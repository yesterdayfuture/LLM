
import torch
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_rel_pos, d_model):
        super().__init__()
        self.emb = nn.Embedding(2 * max_rel_pos + 1, d_model)
    
    def forward(self, seq_len):
        # 生成相对位置索引矩阵（对称）
        rel_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        print(rel_pos)
        # 将相对位置索引映射到嵌入矩阵的索引范围
        rel_pos = torch.clamp(rel_pos + seq_len - 1, 0, 2 * seq_len - 2)
        print(rel_pos)
        
        return self.emb(rel_pos)

# 示例
d_model, max_rel_pos = 64, 10
rel_pe = RelativePositionalEncoding(max_rel_pos, d_model)
rel_enc = rel_pe(seq_len=5)
print("Relative encoding shape:", rel_enc.shape)  # 输出：torch.Size([5, 5, 64])