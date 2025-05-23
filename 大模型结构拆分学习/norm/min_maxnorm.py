import numpy as np

def min_max_normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    print(X_min)
    print(X_max)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled, (X_min, X_max)

# 示例
X = np.array([[1, 2], [3, 4], [5, 6]])
normalized_X, params = min_max_normalize(X)

print("原始数据：")
print(X)
print("归一化后的数据：")
print(normalized_X)
print("归一化参数：")
print(params)