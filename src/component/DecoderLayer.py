# src/component/DecoderLayer.py
import torch.nn as nn
from src.component.SublayerConnection import SublayerConnection
from src.component.MultiHeadAttention import MultiHeadAttention
from src.component.FFN import FFN

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model)
        self.cross_attn = MultiHeadAttention(n_heads, d_model)
        self.ffn = FFN(d_model, d_ff)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Self-attention (with future masking)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Cross-attention
        x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        # FFN
        x = self.sublayer3(x, self.ffn)
        return x