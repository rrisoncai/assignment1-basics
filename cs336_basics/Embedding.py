import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device=None,
            dtype=None
    ):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.W = nn.Parameter(torch.empty(self.vocab_size, self.d_model, **kwargs))
        nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3, b=3)


    def forward(
            self,
            token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.W[token_ids]
    
if __name__ == "__main__":
    vocab_size = 10
    embed_dim = 4
    model = Embedding(vocab_size, embed_dim)
    tokens = torch.tensor([[1,2,3],[4,5,6]])
    out = model(tokens)
    print(out.shape)