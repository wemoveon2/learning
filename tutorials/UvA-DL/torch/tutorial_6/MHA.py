import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from einops import rearrange

def scaled_dot_product(q, k, v, mask=None):
  d_k = q.shape[-1]
  attn_logits = q @ k.transpose(-2, -1) / (d_k ** 0.5)
  if mask is not None:
    attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
  attn_logits = F.softmax(attn_logits, dim=-1)
  values = attn_logits @ v
  return attn_logits, values

def expand_mask(mask):
  assert mask.ndim > 2
  if mask.ndim == 3:
    mask = mask.unsqueeze(1)
  while mask.ndim < 4: 
    mask = mask.unsqueeze(0)
  return mask

class MultiHeadAttention(nn.Module):
  def __init__(self, input_dim, embed_dim, num_heads):
    super().__init__()
    assert embed_dim % num_heads == 0

    self.embed_dim = embed_dim
    self.n_heads = num_heads
    self.head_dim = embed_dim // num_heads
    
    self.qkv_proj =  nn.Linear(input_dim, embed_dim * 3)
    self.output_proj = nn.Linear(embed_dim, embed_dim)
    self._initialize_parameters()
  
  def _initialize_parameters(self):
    nn.init.xavier_uniform_(self.qkv_proj.weight)
    nn.init.xavier_uniform_(self.output_proj.weight)
    self.qkv_proj.bias.data.fill_(0)
    self.output_proj.bias.data.fill_(0)
  
  def forward(self, x, mask=None, return_attention=False):
    n_batch, n_seq, _ = x.shape
    if mask:
      mask = expand_mask(mask)
    qkv = self.qkv_proj(x) # n_batch, n_seq, embed_dim * 3
    # separate q, k, v into 3 vectors of shape n_batch, n_heads, n_seq, embed_dim
    q, k, v = rearrange(qkv, 'b s (h c d) -> c b h s d', c=3, h=self.n_heads)

    attn, values = scaled_dot_product(q, k, v, mask)
    values = rearrange(values, 'b h s d -> b s (h d)')
    o = self.output_proj(values)
    if return_attention:
      return o, attn
    return o
    

