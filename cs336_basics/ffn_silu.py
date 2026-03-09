import torch
from torch import nn
from .util_funcs import silu
class ffn_silu(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=(1.0 / d_model ** 0.5))
        nn.init.trunc_normal_(self.w2, mean=0.0, std=(1.0 / d_model ** 0.5))

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x1 = silu(x @ self.w1.T)
        x4 = x1 @ self.w2.T
        return x4