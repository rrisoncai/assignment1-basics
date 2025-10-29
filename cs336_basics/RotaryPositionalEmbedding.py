import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device=None
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        pos = torch.arange(0, max_seq_len)
        angle = torch.einsum('i,j->ij', pos, inv_freq)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(
            self,
            x: torch.Tensor, # (batch, seq_len, d_model)
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        B, S, D = x.shape
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        x_rot = torch.stack((x1_rot, x2_rot), dim=-1).reshape(B, S, D)
        return x_rot