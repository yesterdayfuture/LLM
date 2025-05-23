'''
Multi-Head Attention 多头注意力

定义​​：将Q/K/V投影到多个子空间，独立计算注意力后拼接。
​​优点​​：捕捉多维度特征、并行计算高效。
​​缺点​​：计算量O(n²d)、显存占用高
'''

import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape

        # 同时计算QKV
        qkv = self.qkv(x).reshape(B, L, 3, -1).permute(2,0,1,3)  # [3, B, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, L, D]
        
        
        # 拆分为多头
        q = q.view(B, L, -1, self.head_dim).transpose(1,2)  # [B, H, L, D]
        k = k.view(B, L, -1, self.head_dim).transpose(1,2)
        v = v.view(B, L, -1, self.head_dim).transpose(1,2)
        
        # 注意力计算
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, L, -1)
        return self.out(out)

# 示例
mha = MHA(embed_dim=512, num_heads=8)
x = torch.randn(2, 10, 512)
print(mha(x).shape)  # torch.Size([2, 10, 512])
