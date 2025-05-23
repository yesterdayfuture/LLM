'''
MQA（Multi-Query Attention，多查询注意力）
原理
    ​​定义​​：所有查询头共享同一组Key/Value。
    ​​优点​​：计算量O(nd)、显存占用减少80%。
    ​​缺点​​：表达能力受限、不适用于复杂任务
'''

import torch
import torch.nn as nn
import math

class MQA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2*self.head_dim)  # 单组KV
        
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, -1, self.head_dim)  # [B, L, H, D]
        kv = self.kv(x).view(B, L, 2, self.head_dim)  # [B, L, 2, D]
        k, v = kv[:,:,0], kv[:,:,1]  # 共享KV

        print(q.shape, k.shape, v.shape)
        print(k.unsqueeze(2).transpose(-2, -1).shape) # 函数 unsqueeze 用法：在指定位置插入一个大小为1的维度
        
        # 注意力计算（广播机制）
        attn = (q @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v.unsqueeze(2)).reshape(B, L, -1)
        return out

# 示例
mqa = MQA(512, 8)
print(mqa(torch.randn(2,10,512)).shape)  # torch.Size([2, 10, 512])

