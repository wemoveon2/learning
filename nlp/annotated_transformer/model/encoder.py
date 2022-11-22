import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union


def clone(module: Union[pl.LightningModule, nn.Module], N: int) -> nn.ModuleList:
    """
    Generates a specified number of clones from the given module.

    Args:
        module (pl.LightningModule): Module/layer to clone.
        N (int): Number of times to clone the given module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Encoder, comprised of a stack of N layers
    """

    def __init__(self, layer: Union[pl.LightningModule, nn.Module], N: int) -> None:
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)  # size is an attribute of the layer class

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Pass input and mask through each layer sequentially

        Args:
            x (torch.Tensor): _description_
            mask (_type_): _description_
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features: torch.Tensor, eps: float=1e-6) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.g * (x - mean) / (std + self.eps) + self.b

class SublayerConnection(nn.Module):
    "Implements residual connection and dropout"

    def __init__(self, size: torch.Tensor, dropout: float=0.5) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Implements the encoder stack"

    def __init__(
        self,
        size: torch.Tensor,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sub_layer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        x = self.sub_layer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sub_layer[1](x, self.feed_forward)
