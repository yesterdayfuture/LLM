'''
GQA（Grouped-Query Attention，分组查询注意力）
原理
    ​​定义​​：将查询头分组，组内共享Key/Value。
    ​​优点​​：平衡MHA和MQA，显存减少30-50%。
    ​​缺点​​：组间信息隔离、需手动调节分组数
'''

import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    def __init__(self, embed_dim, num_heads, groups=4):
        super().__init__()
        assert num_heads % groups == 0, "头数必须可被组数整除"
        self.head_dim = embed_dim // num_heads
        self.groups = groups
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2*self.head_dim*groups)  # 每组KV
        
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.groups, -1, self.head_dim)  # [B, L, G, H/G, D]
        kv = self.kv(x).view(B, L, self.groups, 2, self.head_dim)  # [B, L, G, 2, D]
        k, v = kv[:,:,:,0], kv[:,:,:,1]

        print(q.shape, k.shape, v.shape)
        print(k.unsqueeze(3).shape)
        
        # 组内计算注意力
        attn = (q @ k.unsqueeze(3).transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, L, G, H/G, L]
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v.unsqueeze(3)).reshape(B, L, -1)
        return out

# 示例
gqa = GQA(512, 8, groups=4)
print(gqa(torch.randn(2,10,512)).shape)  # torch.Size([2, 10, 512])


