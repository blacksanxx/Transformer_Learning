import torch
import torch.nn.functional as F
import torch.nn as nn



class FFN(nn.Module):
    def __init__(self,dmodel, dff,dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(dmodel, dff)
        self.linear2 = nn.Linear(dff, dmodel)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 测试代码
if __name__ == "__main__":
    dmodel = 512
    dff = 2048
    ffn = FFN(dmodel, dff)

    # 创建一个随机输入张量，形状为 (batch_size, seq_len, dmodel)
    x = torch.randn(2, 10, dmodel)

    # 前向传播
    output = ffn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)                         
