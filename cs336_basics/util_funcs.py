import torch
import math
from collections.abc import Iterable
import os
import typing

def silu(
        x: torch.Tensor
) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(
        x: torch.Tensor,
        dim: int
) -> torch.Tensor:
    in_type = x.dtype
    x = x.float()
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    out = exp_x / sum_exp_x
    return out.to(in_type)

def softmax_with_temperature(
    x: torch.Tensor,
    temperature: float,
    dim: int
) -> torch.Tensor:
    t = max(temperature, 1e-6)
    return softmax(x / t, dim=dim)

def cross_entropy_loss(
            x: torch.Tensor,
            y: torch.Tensor,
) -> torch.Tensor:
    # Numerically-stable CE on logits with shape (..., vocab) and targets (...,).
    y = y.long()
    x = x.float()
    max_val = x.max(dim=-1, keepdim=True).values
    shifted = x - max_val
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - logsumexp
    nll = -log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    return nll.mean()

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    lr = 0
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        pi = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        lr = min_learning_rate + 0.5 * (1 + math.cos(pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    return lr

def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_l2_norm: float
) -> None:
    eps = 1e-6
    g_norm = 0

    for p in parameters:
        if p.grad is None:
            continue
        g_norm += p.grad.pow(2).sum()
    g_norm = torch.sqrt(g_norm)
    if (g_norm > max_l2_norm):
        scale = max_l2_norm / (g_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad *= scale

import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    total_tokens = dataset.shape[0]
    starts = torch.randint(0, total_tokens - context_length, (batch_size,))
    x = torch.stack([
        torch.from_numpy(dataset[s : s + context_length]) for s in starts
    ])
    y = torch.stack([
        torch.from_numpy(dataset[s + 1 : s + 1 + context_length]) for s in starts
    ])

    x = x.to(device=device)
    y = y.to(device=device)
    return x, y

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    if isinstance(out, (str, os.PathLike)):
        out_dir = os.path.dirname(os.fspath(out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    model_params = model.state_dict()
    optim_params = optimizer.state_dict()
    
    obj = {
        "model": model_params,
        "optim": optim_params,
        "iteration": iteration,
    }
    torch.save(obj, out)
    
def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    # Load checkpoints onto the model's current device so cross-device resumes
    # (e.g., MPS-saved checkpoint resumed on CUDA/CPU) don't fail.
    model_device = next(model.parameters()).device
    obj = torch.load(src, map_location=model_device)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optim"])
    return obj["iteration"]

def nucleus_sampling(
    probs: torch.Tensor,
    prob_cutoff: float,
) -> list[int]:
    # Sort probabilities in descending order and keep original token indices.
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=0)

    # Keep tokens until cumulative probability reaches/exceeds cutoff.
    cutoff_pos = torch.nonzero(cdf >= prob_cutoff, as_tuple=False)
    k = int(cutoff_pos[0].item()) + 1 if cutoff_pos.numel() > 0 else sorted_idx.numel()

    return sorted_idx[:k].tolist()

def decode(
    llm_output: torch.Tensor,
    temperature: float,
    prob_threshold: float,
) -> int:
    # Accept common logits shapes and always decode from the latest position.
    if llm_output.ndim == 1:
        logits = llm_output
    elif llm_output.ndim == 2:
        logits = llm_output[-1]
    elif llm_output.ndim == 3:
        logits = llm_output[0, -1]
    else:
        raise ValueError(f"Unsupported llm_output shape: {tuple(llm_output.shape)}")

    p = min(max(prob_threshold, 0.0), 1.0)
    probs = softmax_with_temperature(logits, temperature, dim=-1)

    nucleus_ids = nucleus_sampling(probs, p)
    nucleus_idx = torch.tensor(nucleus_ids, device=probs.device, dtype=torch.long)
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    sampled_local_idx = torch.multinomial(nucleus_probs, num_samples=1).item()
    token_id = int(nucleus_idx[sampled_local_idx].item())
    return token_id
