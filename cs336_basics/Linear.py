import torch
from torch import nn
import numpy as np

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device=None,
            dtype=None):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, **kwargs))

        delta = np.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=delta * delta, a=-3 * delta, b=3 * delta)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return x @ self.W.T
    
if __name__ == "__main__":
    torch.manual_seed(0)
    layer = Linear(4, 3)
    x = torch.randn(2, 4)
    print(layer(x))