import torch.nn as nn
import torch
import torch
from src.component.scale_dot import scaled_dot_product_attention


"""
多头注意力机制的实现
数学表达：
    MHA(Q, K, V) = Concat(head_1, ..., head_h)W^O

其中每个头：
    head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

"""

class MultiHeadAttention(nn.Module):    
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads    
        self.d_model = d_model    # 词向量维度
        assert d_model % num_heads == 0, "确保均匀分割"
        self.depth = d_model // num_heads   # 每个头的维度 d_v,d_k

       # 线性投影层（注意：这里 Q/K/V 共享输入，但权重不同）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    


    def split_heads(self, x, batch_size):
        # 分割头部并转置以适应注意力计算
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, depth)
    

    def combine_heads(self, x, batch_size):
        # 合并头部
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_heads, depth)
        return x.contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
    

    def forward(self, query, key, value, mask=None):    
        batch_size = query.size(0)

        # 线性投影
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)

        # 分割头部
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 计算缩放点积注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        # 合并头部
        scaled_attention = self.combine_heads(scaled_attention, batch_size)  # (batch_size, seq_len_q, d_model)

        # 最终线性层
        output = self.W_o(scaled_attention)  # (batch_size, seq_len_q, d_model)

        #return output, attention_weights
        return output   # (batch_size, seq_len_q, d_model)


if __name__ == "__main__":
    # 测试多头注意力机制
    num_heads = 2
    d_model = 4
    mha = MultiHeadAttention(num_heads, d_model)

    # 创建测试输入 (batch_size, seq_len, d_model)
    query = torch.rand(1, 3, d_model)
    key = torch.rand(1, 3, d_model)
    value = torch.rand(1, 3, d_model)

    output, attn_weights = mha.forward(query, key, value)
    print("Multi-Head Attention Output:")
    print(output)
    print("Attention Weights:")
    print(attn_weights)

