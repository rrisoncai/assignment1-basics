import torch
import math
from collections.abc import Iterable

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
    # print(f"input.shape={x.shape}, output.shape={y.shape}")
    # print(f"x={x}")
    # print(f"y={y}")
    # print(f"x[:,y]={x[y]}")
    max_val = x.max(dim=-1, keepdim=True).values
    # print(f"max_val={max_val}")
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