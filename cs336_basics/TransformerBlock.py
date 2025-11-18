import torch
from torch import nn
from .rmsnorm import rmsnorm
from .CausalMultiheadSelfAttention import CausalMultiheadSelfAttention
from .swiglu import swiglu

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: int,
    ):
        super().__init__()
        self.rmsnorm1 = rmsnorm(d_model=d_model)
        self.rmsnorm2 = rmsnorm(d_model=d_model)
        self.sdpa = CausalMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )

        self.ffn = swiglu(d_model, d_ff)
        
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x1 = x + self.sdpa(self.rmsnorm1(x))
        x2 = x1 + self.ffn(self.rmsnorm2(x1))
        return x2