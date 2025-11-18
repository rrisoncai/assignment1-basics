"""
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of training and validation large datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).
"""

import argparse
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.util_funcs import *  # cross_entropy_loss, save_checkpoint, load_checkpoint, etc.

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class MemmapDataset(Dataset):
    def __init__(self, path, context_length: int):
        self.data = np.memmap(path, mode="r", dtype=np.int16)
        self.context_length = context_length
        self.num_tokens = len(self.data)
        self.num_possible = self.num_tokens - context_length

    def __len__(self):
        return self.num_possible

    def __getitem__(self, index):
        # x: first context_length tokens, y: next tokens shifted by 1
        x = self.data[index : index + self.context_length]
        y = self.data[index + 1 : index + 1 + self.context_length]

        x = torch.from_numpy(np.array(x, copy=True)).long()
        y = torch.from_numpy(np.array(y, copy=True)).long()
        return x, y


def make_dataloader(path, context_length, batch_size, shuffle=True):
    ds = MemmapDataset(path, context_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="/Users/dawn/ws/stanford-cs336/data/token_ids/TinyStoriesV2-GPT4-train.txt.npy")
    parser.add_argument("--val_path", type=str, default="/Users/dawn/ws/stanford-cs336/data/token_ids/TinyStoriesV2-GPT4-valid.txt.npy")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=4096)
    parser.add_argument("--theta", type=int, default=10_000)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1_000)
    parser.add_argument("--ckpt_path", type=str, default="ckpts/latest.pt")

    return parser.parse_args()


def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)  # your CE implementation
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Available Deivce: {device}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )

    train_loader = make_dataloader(args.train_path, args.context_length, args.batch_size)
    print(f"Loaded train dataset: total tokens={train_loader.dataset.num_tokens}, steps per epoch={len(train_loader)}")
    
    val_loader = make_dataloader(args.val_path, args.context_length, args.batch_size)
    print(f"Loaded val dataset: total tokens={val_loader.dataset.num_tokens}, steps per epoch={len(val_loader)}")

    start_step = 0
    if os.path.exists(args.ckpt_path):
        start_step = load_checkpoint(args.ckpt_path, model, optimizer, device)
        print(f"Resumed from checkpoint '{args.ckpt_path}' at step {start_step}")

    train_iter = iter(train_loader)
    print("Starting training...")
    print(f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, steps={args.max_steps}")

    for step in range(start_step, args.max_steps):
        step_start = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        if step_start is None:
            import time
            cpu_t0 = time.time()
        else:
            step_start.record()

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        if (step + 1) % 1000 == 0:
            print(f"[debug] step {step+1}: x.shape={x.shape}, logits.shape={logits.shape}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            step_end = torch.cuda.Event(enable_timing=True)
            step_end.record()
            torch.cuda.synchronize()
            step_ms = step_start.elapsed_time(step_end)
            print(f"[time] step {step+1}: {step_ms:.2f} ms")
        else:
            import time
            step_ms = (time.time() - cpu_t0) * 1000
            print(f"[time] step {step+1}: {step_ms:.2f} ms")

        if (step + 1) % args.log_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"step {step + 1}: train loss = {loss.item():.4f}, val loss = {val_loss:.4f}")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(args.ckpt_path, model, optimizer, step + 1)

if __name__ == "__main__":
    main()