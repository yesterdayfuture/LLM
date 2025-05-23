import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        return x + self.pe(positions)

# 示例
d_model, max_len = 64, 50
inputs = torch.randn(32, max_len, d_model)  # 模拟输入 (batch_size=32, seq_len=50)
pe_layer = LearnablePositionalEncoding(max_len, d_model)
output = pe_layer(inputs)
print("Encoded shape:", output.shape)  # 输出：torch.Size([32, 50, 64])