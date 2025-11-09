import torch.nn as nn
import torch
from src.component.PositionalEncoding import PositionalEncoding
from src.component.EncoderLayer import EncoderLayer
from src.component.DecoderLayer import DecoderLayer

def _generate_future_mask(sz):
    """生成 future mask（下三角为 True）"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask  # [sz, sz]

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.final_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.tgt_embed(tgt) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.final_proj(out)  # [B, T, tgt_vocab]
    def generate(self, src, src_mask, max_len=50, sos_idx=1, eos_idx=2, pad_idx=0):
        """
        自回归生成目标序列
        src: [1, src_len]
        src_mask: [1, src_len]
        """
        self.eval()
        with torch.no_grad():
            # 编码源句子
            memory = self.encode(src, src_mask)  # 假设你有 encode 方法

            # 初始化目标序列: [<sos>]
            ys = torch.full((1, 1), sos_idx, dtype=torch.long, device=src.device)

            for _ in range(max_len - 1):
                # 生成当前目标 mask（因果 mask）
                tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(src.device)
                tgt_padding_mask = (ys != pad_idx).unsqueeze(1).unsqueeze(2)  # [1,1,1,cur_len]

                # 解码
                out = self.decode(ys, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=~tgt_padding_mask.squeeze(1))
                prob = self.generator(out[:, -1, :])  # 取最后一个 token 的 logits
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()

                ys = torch.cat([ys, torch.full((1, 1), next_word, dtype=torch.long, device=src.device)], dim=1)

                if next_word == eos_idx:
                    break

            return ys.squeeze(0)  # [seq_len]