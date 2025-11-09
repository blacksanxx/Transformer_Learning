import torch.nn as nn
from src.component.SublayerConnection import SublayerConnection
from src.component.MultiHeadAttention import MultiHeadAttention
from src.component.FFN import FFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads,d_model)
        self.ffn = FFN(d_model, d_ff)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        # FFN with residual
        x = self.sublayer2(x, self.ffn)
        return x