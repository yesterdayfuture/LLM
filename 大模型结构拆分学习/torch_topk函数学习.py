
import torch
# 生成随机数据（3个样本，5个类别）
logits = torch.randn(3, 5)

# 获取前3个最大值及其索引
top_values, top_indices = torch.topk(logits, k=3, dim=-1)

print("原始张量:\n", logits)
print("前3个值:\n", top_values) #shape 为 (样本数 或 批次大小*序列长度， k)
print("对应索引:\n", top_indices) #shape 为 (样本数 或 批次大小*序列长度， k)