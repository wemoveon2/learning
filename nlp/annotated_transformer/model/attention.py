import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from model.encoder import clone

"""
The attention function is a mapping of a query and key-value pair to an output.

The output is the weighted sum of the values, where the weight assigned to the 
value is determined by a compatibility function of the query with the corresponding
key.

Scaled Dot Product Attention

Attention(Q,K,V) = softmax(Q * K^T / sqrt(dk)) * V

Where dk = dq

- There are dot product and additive attention, dot product preferred as the
operations can be vectorized into matrix multiplication.
- Additive outperforms dot product attention without scaling for large values 
of dk.
"""

def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def dot_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None, dropout=None) -> Tuple[torch.Tensor, torch.Tensor]:
    dk = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(dk)
    if mask is not None: # apply masking to future tokens if provided
        scores = scores.masked_fill(mask == 0, -1e9) # fill future tokens at each timestep with large neg number
    scores = F.softmax(scores, dim=-1) # Convert to probabilities
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, value), scores


class MultiHeadedAttention(nn.Module):

    def __init__(self, heads:int, d_model:int, p_dropout:float=0.1) -> None:
        super().__init__()
        assert d_model % heads == 0, "number of heads must be factor of the model's dimensions d_m"
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=p_dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        # linearly project from d_model -> h x d_k
        query, key, value = [lin(x).view(n_batches, -1, self.heads, self.d_k).transpose(1,2) for lin, x in zip(self.linears, (query, key, value))]
        # apply attention on the project vectors
        attn, self.attn = dot_attention(query, key, value, mask, self.dropout)
        # concat and apply final linear projection
        x = x.transpose(1,2).contiguous.view(n_batches, -1, self.heads*self.d_k)
        del query, key, value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, p_dropout: float=0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

