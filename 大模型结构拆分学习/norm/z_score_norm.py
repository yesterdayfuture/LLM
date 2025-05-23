'''
原理​​：调整为均值为0、标准差为1的分布
'''

import numpy as np

def z_score_standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_scaled = (X - mu) / sigma
    return X_scaled, (mu, sigma)

# 示例
X = np.array([[1, 2], [3, 4], [5, 6]])
standardized_X, params = z_score_standardize(X)

print("标准化后的数据：\n", standardized_X)
print("均值和标准差：", params)