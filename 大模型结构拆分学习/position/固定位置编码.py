

import numpy as np
import matplotlib.pyplot as plt

# np.newaxis = None 用于占位，增加维度
print(np.newaxis)

def sinusoidal_position_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis] # np.newaxis用于增加维度
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# 示例
max_len, d_model = 50, 64
pe = sinusoidal_position_encoding(max_len, d_model)

# 可视化
plt.imshow(pe, cmap='viridis', aspect='auto')
plt.title("Sinusoidal Position Encoding")
plt.colorbar()
plt.show()
