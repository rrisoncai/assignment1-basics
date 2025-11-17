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
            weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.rmsnorm1 = rmsnorm(d_model=d_model)
        self.rmsnorm1.load_state_dict({"weight": weights["ln1.weight"]})
        
        self.rmsnorm2 = rmsnorm(d_model=d_model)
        self.rmsnorm2.load_state_dict({"weight": weights["ln2.weight"]})


        self.sdpa = CausalMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        state = {
            "Q": weights["attn.q_proj.weight"],
            "K": weights["attn.k_proj.weight"],
            "V": weights["attn.v_proj.weight"],
            "O": weights["attn.output_proj.weight"],
        }
        self.sdpa.load_state_dict(state)

        self.ffn = swiglu(d_model, d_ff)
        state = {
            "w1": weights["ffn.w1.weight"],
            "w2": weights["ffn.w2.weight"],
            "w3": weights["ffn.w3.weight"]
        }
        self.ffn.load_state_dict(state)
        
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x1 = x + self.sdpa(self.rmsnorm1(x))
        x2 = x1 + self.ffn(self.rmsnorm2(x1))
        return x2