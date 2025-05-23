

'''
仅用均方根替代标准差，省去均值中心化
RMS(x)= sqrt(∑ x**2/n)
y= x / ( RMS(x)+ϵ ) ⋅ γ
'''

import numpy as np

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones(dim)

    def forward(self, x):
        rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
        return x / rms * self.gamma

# 示例
X = np.array([[1, 2, 3], [4, 5, 6]])

rms_norm = RMSNorm(dim=3)
output = rms_norm.forward(X)

print(output)


