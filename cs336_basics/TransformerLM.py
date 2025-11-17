import torch
from torch import nn
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock
from .rmsnorm import rmsnorm
from .Linear import Linear
from .util_funcs import softmax

def get_layer_weights(weights: dict[str, torch.Tensor], layer_idx: int):
    prefix = f"layers.{layer_idx}."
    layer_w = {k.replace(prefix, ""): v for k, v in weights.items() if k.startswith(prefix)}
    return layer_w

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
            weights: dict[str, torch.Tensor],
            token_positions: torch.Tensor,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.embed.load_state_dict({"W": weights["token_embeddings.weight"]})

        self.transformer_stack = nn.ModuleList(
            [TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                theta,
                get_layer_weights(weights, i),
                token_positions,
            )
            for i in range(num_layers)]
        )

        self.norm = rmsnorm(d_model)
        self.norm.load_state_dict({"weight": weights["ln_final.weight"]})

        self.linear = Linear(d_model, vocab_size)
        self.linear.load_state_dict({"W": weights["lm_head.weight"]})

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