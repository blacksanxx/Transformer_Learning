import torch
import torch.nn as nn

class SublayerConnection(nn.Module):
    """
    残差连接 + LayerNorm
    """
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: 输入张量 [batch_size, seq_len, d_model]
        sublayer: 一个函数（如 MultiHeadAttention 或 FFN）
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
# ========== 测试代码 ==========
if __name__ == "__main__":
    from FFN import FFN 

    # 测试 SublayerConnection
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    ffn = FFN(d_model, dff=2048)
    sublayer_conn = SublayerConnection(d_model, dropout=0.1)
    
    out = sublayer_conn(x, ffn)
    print("✅ SublayerConnection output shape:", out.shape)
    assert out.shape == x.shape, "Shape mismatch!"
    print("✅ SublayerConnection test passed!")