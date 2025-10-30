import torch

def softmax(
        x: torch.Tensor,
        dim: int
) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x