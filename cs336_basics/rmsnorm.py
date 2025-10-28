import torch
from torch import nn
import numpy as np
from einops import rearrange, reduce

class rmsnorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        scale = torch.sqrt(reduce(x ** 2, 'b s d -> b s 1', 'mean') + self.eps)
        result = self.weight * x / scale
        return result.to(in_dtype)