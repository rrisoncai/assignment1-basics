"""
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of training and validation large datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).
"""

import argparse
import contextlib
from datetime import datetime
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.util_funcs import *  # cross_entropy_loss, save_checkpoint, load_checkpoint, etc.

import numpy as np
import os
import torch
import wandb
from torch.utils.data import Dataset, DataLoader


class MemmapDataset(Dataset):
    def __init__(self, path, context_length: int):
        # Load .npy correctly via NumPy so we don't treat file headers as tokens.
        self.data = np.load(path, mmap_mode="r")
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
    parser.add_argument("--theta", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Training precision. On CUDA, bf16/fp16 use autocast to reduce memory.",
    )
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1_000)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, use ckpts/<timestamp>/latest.pt and train from scratch.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="wandb mode. Use 'offline' to avoid network upload during training.",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=200,
        help="Max number of validation batches to run per evaluation (-1 = full val set).",
    )

    return parser.parse_args()


def evaluate(model, val_loader, device, max_batches: int = -1):
    model.eval()
    total_loss, total_batches = 0.0, 0

    amp_dtype = None
    if device.type == "cuda":
        if getattr(model, "_train_precision", "fp32") == "bf16":
            amp_dtype = torch.bfloat16
        elif getattr(model, "_train_precision", "fp32") == "fp16":
            amp_dtype = torch.float16

    with torch.inference_mode():
        for x, y in val_loader:
            if max_batches > 0 and total_batches >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                logits = model(x)
                loss = cross_entropy_loss(logits, y)  # your CE implementation
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_steps

    if not args.ckpt_path:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("ckpts", run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        args.ckpt_path = os.path.join(run_dir, "latest.pt")
        print(f"No --ckpt_path provided. Using timestamped checkpoint path: {args.ckpt_path}")
    else:
        ckpt_dir = os.path.dirname(args.ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
    model._train_precision = args.precision

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))

    train_loader = make_dataloader(args.train_path, args.context_length, args.batch_size)
    print(f"Loaded train dataset: total tokens={train_loader.dataset.num_tokens}, steps per epoch={len(train_loader)}")
    
    val_loader = make_dataloader(args.val_path, args.context_length, args.batch_size)
    print(f"Loaded val dataset: total tokens={val_loader.dataset.num_tokens}, steps per epoch={len(val_loader)}")

    start_step = 0
    if os.path.exists(args.ckpt_path):
        start_step = load_checkpoint(args.ckpt_path, model, optimizer)
        print(f"Resumed from checkpoint '{args.ckpt_path}' at step {start_step}")
    else:
        print(f"No checkpoint found at '{args.ckpt_path}'. Starting from scratch.")

    train_iter = iter(train_loader)
    print("Starting training...")
    print(f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, steps={args.max_steps}")

    wandb_enabled = args.use_wandb and args.wandb_mode != "disabled"
    if wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

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

        amp_dtype = None
        if device.type == "cuda":
            if args.precision == "bf16":
                amp_dtype = torch.bfloat16
            elif args.precision == "fp16":
                amp_dtype = torch.float16
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(x)
            loss = cross_entropy_loss(logits, y)

        lr = learning_rate_schedule(
            it=step,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if device.type == 'cuda':
            step_end = torch.cuda.Event(enable_timing=True)
            step_end.record()
            torch.cuda.synchronize()
            step_ms = step_start.elapsed_time(step_end)
            # print(f"[time] step {step+1}: {step_ms:.2f} ms")
        else:
            import time
            step_ms = (time.time() - cpu_t0) * 1000
            # print(f"[time] step {step+1}: {step_ms:.2f} ms")

        if (step + 1) % args.log_interval == 0:
            val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
            print(f"step {step + 1}: train loss = {loss.item():.4f}, val loss = {val_loss:.4f}")
            if wandb_enabled:
                tokens_per_step = args.batch_size * args.context_length
                tokens_per_sec = tokens_per_step / max(step_ms / 1000.0, 1e-8)
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/loss": val_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "perf/step_ms": step_ms,
                        "perf/tokens_per_sec": tokens_per_sec,
                    },
                    step=step + 1,
                )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, step + 1, args.ckpt_path)

    if wandb_enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
