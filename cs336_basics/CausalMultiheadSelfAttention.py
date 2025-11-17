import torch
from torch import nn
from .scaled_dp_attn import scaled_dp_attn
from einops import einsum, rearrange, repeat

class CausalMultiheadSelfAttention(nn.Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            theta: int | None = None,
            max_seq_len: int | None = None,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Q = nn.Parameter(torch.empty(d_model, d_model))
        self.K = nn.Parameter(torch.empty(d_model, d_model))
        self.V = nn.Parameter(torch.empty(d_model, d_model))
        self.O = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.trunc_normal_(self.Q, mean=0.0, std=(1.0 / self.d_model ** 0.5))
        nn.init.trunc_normal_(self.K, mean=0.0, std=(1.0 / self.d_model ** 0.5))
        nn.init.trunc_normal_(self.V, mean=0.0, std=(1.0 / self.d_model ** 0.5))
        nn.init.trunc_normal_(self.O, mean=0.0, std=(1.0 / self.d_model ** 0.5))

        if theta is not None and max_seq_len is not None:
            from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        B, S, D = x.shape
        H = self.num_heads
        assert D % H == 0, "d_model must be divisible by num_heads"
        d_k = D // H
        Q = rearrange(x @ self.Q.T, "b s (h d) -> b h s d", h=H, d=d_k)
        K = rearrange(x @ self.K.T, "b s (h d) -> b h s d", h=H, d=d_k)
        V = rearrange(x @ self.V.T, "b s (h d) -> b h s d", h=H, d=d_k)

        if self.rope is not None:
            seq_len = Q.shape[-2]
            token_positions = torch.arange(seq_len, device=Q.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = ~torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
        mask = repeat(mask, 's1 s2 -> b h s1 s2', b=B, h=H)
        sdpa = scaled_dp_attn(Q, K, V, mask)
        out = sdpa.transpose(1,2).contiguous().view(B, S, D)
        return out @ self.O.T