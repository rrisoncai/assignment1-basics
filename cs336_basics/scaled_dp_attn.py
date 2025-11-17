import torch
from .util_funcs import softmax
from einops import einsum

def scaled_dp_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
        Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    scores = softmax(scores, dim=-1)
    attn = einsum(scores, V, "... q v, ... v d_v -> ... q d_v")
    return attn