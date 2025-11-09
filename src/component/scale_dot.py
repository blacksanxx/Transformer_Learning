import torch
import torch.nn.functional as F
import math

"""
1.0版本的代码

实现了缩放点积注意力机制（Scaled Dot-Product Attention）。
伪代码:
    输入: Q (n x dk), K (m x dk), V (m x dv)
    计算: scores = Q @ K^T / sqrt(dk)
    计算: attention_weights = softmax(scores, dim=-1)
    输出: output = attention_weights @ V

# 缩放点积注意力机制实现
def scaled_dot_product_attention(query, key, value, mask=None):
    dk = query.size(-1)
    # 计算 QK^T / sqrt(dk)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        
        #src_mask (形状 [B, T_k]) 
        #tgt_mask (形状 [T_q, T_k] 或 [T_tgt, T_tgt])
        
        #print("mask shape:", mask.shape) // 这里 mask shape: torch.Size([32, 128]) [batch_size, seq_len]
        # score:[batch_size, num_heads, seq_len_q, seq_len_k]
        print("pre；==================== mask shape:", mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len_k)
        print("post==================== mask shape:", mask.shape)
        print("scores shape:", scores.shape)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
"""


# 2.0 --- src和target mask 形状不一致兼容
def scaled_dot_product_attention(query, key, value, mask=None):
    dk = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        B, n_heads, seq_len_q_in_scores, seq_len_k_in_scores = scores.shape
        B_mask, T_k_mask = mask.shape

        if B_mask == B and T_k_mask == seq_len_k_in_scores: # mask is [B, seq_len_k] (e.g., src_mask)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, seq_len_k]
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        elif mask.shape == (seq_len_q_in_scores, seq_len_k_in_scores): # mask is [seq_len_q, seq_len_k] (e.g., tgt_mask)
            mask_expanded = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask_expanded, float('-inf')) # tgt_mask is boolean
        else:
            raise ValueError(f"Mask shape {mask.shape} is not compatible with scores shape {scores.shape}")

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights








# 测试代码
if __name__ == "__main__":
    # 测试没有batch_size的情况
    Q = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    K = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    V = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
    output = scaled_dot_product_attention(Q, K, V)
    print("Output without batch size:")
    print(output)

    # 测试具有batch_size的情况
    Q_batch = torch.tensor([[[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
                            [[1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0]]])
    K_batch = torch.tensor([[[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
                            [[1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0]]])
    V_batch = torch.tensor([[[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0]],
                            [[7.0, 8.0],
                             [9.0, 10.0],
                             [11.0, 12.0]]])
    output_batch = scaled_dot_product_attention(Q_batch, K_batch, V_batch)
    print("Output with batch size:")
    print(output_batch)