'''
原理​​：对每个特征通道在批次维度归一化
'''

import numpy as np


class BatchNorm1D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        1D 批量归一化层
        
        参数：
            num_features: 输入特征维度（每个特征单独归一化）
            eps: 数值稳定性系数，防止除零错误（默认 1e-5）
            momentum: 移动平均系数，用于更新全局统计量（默认 0.1）
        """
        # 可学习的缩放参数 gamma（初始化为全1向量）
        self.gamma = np.ones(num_features)
        # 可学习的偏移参数 beta（初始化为全0向量）
        self.beta = np.zeros(num_features)
        # 运行时统计量：训练过程中逐步更新的全局均值
        self.running_mean = np.zeros(num_features)
        # 运行时统计量：训练过程中逐步更新的全局方差
        self.running_var = np.ones(num_features)  # 初始化为1避免初始除零错误
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, training=True):
        """
        前向传播
        
        参数：
            x: 输入数据，形状为 (batch_size, num_features)
            training: 是否为训练模式（决定是否更新运行时统计量）
        """
        if training:
            # --- 训练阶段 ---
            # 计算当前批次的均值（沿batch维度计算，保持特征独立性）
            mu = x.mean(axis=0)
            # 计算当前批次的方差（无偏估计，分母为batch_size - 1）
            var = x.var(axis=0)
            
            # --- 关键：更新全局移动平均 ---
            # 使用指数移动平均（EMA）更新全局均值：
            # running_mean = momentum *历史均值 + (1 - momentum) * 当前均值
            # 目的是在训练过程中逐步积累全局统计量，替代简单的算术平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            
            # 同理更新全局方差（注意：这里使用有偏方差估计，与PyTorch实现一致）
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # --- 推理阶段 ---
            # 直接使用训练阶段积累的全局统计量，不更新参数
            mu = self.running_mean
            var = self.running_var

        # 标准化：x_hat = (x - μ) / sqrt(σ² + ε)
        x_hat = (x - mu) / np.sqrt(var + self.eps)  # self.eps防止除零
        
        # 缩放和平移：y = gamma * x_hat + beta
        return self.gamma * x_hat + self.beta


# 示例

X = np.array([[1, 2], [3, 4], [5, 6]])
bn = BatchNorm1D(num_features=2)
output = bn.forward(X, training=True)

print(output)


'''
使用pytorch实现
'''
import torch
import torch.nn as nn

# 定义批量归一化层
batch_norm = nn.BatchNorm1d(num_features=2)

print(batch_norm(torch.tensor(X, dtype=torch.float32)))