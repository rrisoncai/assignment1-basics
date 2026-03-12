import torch
from torch import nn
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock
from .rmsnorm import rmsnorm
from .Linear import Linear
from .util_funcs import softmax

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: int,
            weight_tying: bool = False,
            weights: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)

        self.transformer_stack = nn.ModuleList(
            [TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                theta,
            )
            for i in range(num_layers)]
        )

        self.norm = rmsnorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        if weight_tying:
            # Tie output projection to token embedding weights:
            # logits = h @ W_embed^T
            self.linear.W = self.embed.W

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(x)
        for block in self.transformer_stack:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        # out = softmax(x, dim=-1)
        return x
