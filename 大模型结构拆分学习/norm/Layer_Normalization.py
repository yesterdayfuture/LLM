'''
原理​​：对单个样本所有特征归一化
'''
import numpy as np

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):

        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        print('mu:', mu)
        print('var:', var)
        # 对每个样本的所有特征进行归一化
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# 示例（输入维度：[batch, features]）
X = np.array([[1.0, 2.0], [3.0, 4.0]])
ln = LayerNorm(normalized_shape=2)
output = ln.forward(X)

print(output)


'''
使用pytorch实现
'''
import torch
import torch.nn as nn

# 定义层归一化层
batch_norm = nn.LayerNorm(normalized_shape=2)

print(batch_norm(torch.tensor(X, dtype=torch.float32)))
