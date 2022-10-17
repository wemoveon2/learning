import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn


class EncoderDecoder(pl.LightningModule):
    """
    Standard encoder decoder architecture.

    Args:
        pl (_type_): _description_
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Processes masked source and target sequences.

        Args:
            src (_type_): _description_
            tgt (_type_): _description_
            src_mask (_type_): _description_
            tgt_mask (_type_): _description_
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(pl.LightningModule):
    """Define standard linear + softmax generation step.

    Args:
        pl (_type_): _description_
    """

    def __init__(self, d_model, vocab) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
