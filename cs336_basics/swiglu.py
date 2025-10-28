import torch
from torch import nn

class swiglu(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=(1.0 / d_model ** 0.5))
        nn.init.trunc_normal_(self.w2, mean=0.0, std=(1.0 / d_model ** 0.5))
        nn.init.trunc_normal_(self.w3, mean=0.0, std=(1.0 / d_model ** 0.5))

    def silu(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return x * torch.sigmoid(x)
        
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.silu(x @ self.w1.T)
        x2 = x @ self.w3.T
        x3 = x1 * x2
        x4 = x3 @ self.w2.T
        return x4