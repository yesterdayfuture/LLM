
import torch
import math

def rotate_half(x):
    #x1, x2 = x.chunk(2, dim=-1)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, freq):
    cos, sin = freq.cos(), freq.sin()
    print(cos.shape, sin.shape)
    cos, sin = cos.repeat(2), sin.repeat(2)
    print(cos.shape, sin.shape)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

# 示例
d_model, seq_len = 64, 50
q = torch.randn(1, seq_len, d_model)
k = torch.randn(1, seq_len, d_model)
freq = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
q_rot, k_rot = apply_rotary_emb(q, k, freq)
print("Rotated shapes:", q_rot.shape, k_rot.shape)  # 输出：torch.Size([1, 50, 64])



