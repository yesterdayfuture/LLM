深度学习中常见的位置编码方式及其Python实现：

---

一、**固定位置编码（Sinusoidal Positional Encoding）**
原理
通过不同频率的正弦和余弦函数生成位置编码，使模型能够捕捉绝对位置和相对位置信息。公式为：
\[
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad 
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
其中 `pos` 是位置索引，`i` 是维度索引，`d_model` 是模型维度。

Python实现
```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_position_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
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
```
输出示例：生成一个形状为 `(50, 64)` 的编码矩阵，低频维度变化平缓，高频维度变化剧烈。

---

二、**可学习位置编码（Learnable Positional Encoding）**
原理
将位置编码作为可训练参数，通过嵌入层动态学习每个位置的表示。

Python实现（PyTorch）
```python
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
```
优势：灵活性高，适合特定任务；缺点：依赖预定义的最大序列长度。

---

三、**相对位置编码（Relative Positional Encoding）**
原理
关注序列元素之间的相对位置差异，常用于长序列建模。

Python实现
```python
class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_rel_pos, d_model):
        super().__init__()
        self.emb = nn.Embedding(2 * max_rel_pos + 1, d_model)
    
    def forward(self, seq_len):
        # 生成相对位置索引矩阵（对称）
        rel_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        rel_pos = torch.clamp(rel_pos + seq_len - 1, 0, 2 * seq_len - 2)
        return self.emb(rel_pos)

# 示例
d_model, max_rel_pos = 64, 10
rel_pe = RelativePositionalEncoding(max_rel_pos, d_model)
rel_enc = rel_pe(seq_len=5)
print("Relative encoding shape:", rel_enc.shape)  # 输出：torch.Size([5, 5, 64])
```
应用场景：Transformer-XL、音乐生成等长序列任务。

---

四、**旋转位置编码（Rotary Positional Encoding, RoPE）**
原理
通过旋转矩阵将绝对位置信息融入注意力计算，保持相对位置的线性性质。
数学公式
\[
R_{\theta,m} = \begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{bmatrix}, \quad 
q' = R_{\theta,m}q, \quad k' = R_{\theta,n}k
\]
内积结果为 \( q^T R_{m-n}k \)，自然包含相对位置信息。

Python实现
```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, freq):
    cos, sin = freq.cos(), freq.sin()
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
```
优势：支持任意长度外推，广泛用于LLaMA、ChatGLM等大模型。

---

五、**多尺度位置编码（Multi-scale Positional Encoding）**
原理
在不同尺度上编码位置信息，增强模型对局部和全局结构的感知。
```python
class MultiScalePositionalEncoding(nn.Module):
    def __init__(self, d_model, scales=[100, 1000]):
        super().__init__()
        self.scales = scales
        self.embs = nn.ModuleList([nn.Embedding(s, d_model) for s in scales])
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        encodings = [emb(positions % s) for emb, s in zip(self.embs, self.scales)]
        return x + sum(encodings)

# 示例
ms_pe = MultiScalePositionalEncoding(d_model=64, scales=[50, 100])
output = ms_pe(torch.randn(32, 100, 64))
print("Multi-scale output shape:", output.shape)  # 输出：torch.Size([32, 100, 64])
```

---

**总结与选择建议**
| 方法               | 适用场景                     | 优点                          | 缺点                          |
|--------------------|-----------------------------|-------------------------------|-------------------------------|
| 固定位置编码       | 通用NLP任务                 | 确定性，无需训练               | 无法自适应长序列               |
| 可学习位置编码      | 短序列任务                  | 灵活性高                      | 依赖预定义长度，泛化性差       |
| 相对位置编码        | 长文本生成、音乐建模         | 捕捉相对位置关系              | 计算复杂度较高                |
| 旋转位置编码        | 大语言模型（LLaMA等）       | 支持外推，数学性质优雅         | 实现较复杂                   |
| 多尺度编码          | 多粒度任务（如蛋白质结构）   | 兼顾局部和全局信息            | 参数较多，需调参              |

完整代码库可参考网页。实际应用中需根据任务特点选择编码方式，例如Transformer推荐固定编码，大模型优先RoPE。