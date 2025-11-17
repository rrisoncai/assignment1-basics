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
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def cross_entropy_loss(
            x: torch.Tensor,
            y: torch.Tensor,
) -> torch.Tensor:

    max_val = x.max(dim=-1, keepdim=True).values
    x = x - max_val
    true_labels = x.gather(1, y.unsqueeze(1)).squeeze(1)

    exp_x = torch.exp(x)
    logsum_exp_x = torch.log(exp_x.sum(dim=-1, keepdim=True))
    loss = logsum_exp_x - true_labels
    loss = loss.mean()
    return loss

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
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optim"])
    return obj["iteration"]