'''
MLA（Multi-Head Latent Attention，多头潜在注意力）
原理
    ​​定义​​：通过低秩压缩Key/Value到潜在空间。
    ​​优点​​：显存减少93%、支持超长序列。
    ​​缺点​​：压缩可能损失信息、需额外训练投影矩阵
'''

import torch
import torch.nn as nn
import math

class MLA(nn.Module):
    def __init__(self, embed_dim, latent_dim=64, num_heads=8):
        super().__init__()
        self.latent_proj = nn.Linear(embed_dim, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads)
        self.out_proj = nn.Linear(latent_dim, embed_dim)
        
    def forward(self, x):
        latent = self.latent_proj(x)  # 降维到潜在空间
        print("latent.shape ", latent.shape)

        latent = latent.permute(1,0,2)  # [L, B, D]
        print("latent.permute(1,0,2).shape ", latent.shape)

        attn_out, _ = self.attn(latent, latent, latent)
        print("attn_out.shape ", attn_out.shape)
        
        return self.out_proj(attn_out.permute(1,0,2))

# 示例
mla = MLA(512)
print(mla(torch.randn(2,10,512)).shape)  # torch.Size([2, 10, 512])
