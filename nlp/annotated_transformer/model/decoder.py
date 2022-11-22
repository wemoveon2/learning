from numpy import diag
import torch
import torch.nn as nn
from encoder import clone, LayerNorm, SublayerConnection


class Decoder(nn.Module):
    "N layer decoder with masking"

    def __init__(self, layer: nn.Module, N: int) -> None:
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder layer implements self attn, src attn, and a feed forward as sublayers"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)



