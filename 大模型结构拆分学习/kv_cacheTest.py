import torch
import torch.nn as nn

# 超参数
d_model = 4
n_heads = 1
seq_len = 3
batch_size = 3

# 初始化参数（兼容多头形式）
Wq = nn.Linear(d_model, d_model, bias=False)
Wk = nn.Linear(d_model, d_model, bias=False)
Wv = nn.Linear(d_model, d_model, bias=False)

# 生成模拟输入（整个序列一次性输入）
input_sequence = torch.randn(batch_size, seq_len, d_model)  # [B, L, D]

# 初始化 KV 缓存（兼容多头格式）
kv_cache = {
    "keys": torch.empty(batch_size, 0, n_heads, d_model // n_heads),  # [B, T, H, D/H]
    "values": torch.empty(batch_size, 0, n_heads, d_model // n_heads) 
}

# 因果掩码预先生成（覆盖最大序列长度）
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # [L, L]


'''
本循环是将整句话中的token一个一个输入，并更新KV缓存；
所以无需显示的因果掩码，因为因果掩码只用于计算注意力权重时，而计算注意力权重时，KV缓存中的key和value已经包含了因果掩码的信息。

'''


for step in range(seq_len):
    # 1. 获取当前时间步的输入（整个批次）
    current_token = input_sequence[:, step, :]  # [B, 1, D]
    
    # 2. 计算当前时间步的 Q/K/V（保持三维结构）
    q = Wq(current_token)  # [B, 1, D]
    k = Wk(current_token)  # [B, 1, D]
    v = Wv(current_token)  # [B, 1, D]
    
    # 3. 调整维度以兼容多头格式（关键修改点）
    def reshape_for_multihead(x):
        return x.view(batch_size, 1, n_heads, d_model // n_heads).transpose(1, 2)  # [B, H, 1, D/H]
    
    # 4. 更新 KV 缓存（增加时间步维度）
    kv_cache["keys"] = torch.cat([
        kv_cache["keys"], 
        reshape_for_multihead(k).transpose(1, 2)  # [B, T+1, H, D/H]
    ], dim=1)
    
    kv_cache["values"] = torch.cat([
        kv_cache["values"],
        reshape_for_multihead(v).transpose(1, 2)  # [B, T+1, H, D/H]
    ], dim=1)
    
    # 5. 多头注意力计算（支持批量处理）
    q_multi = reshape_for_multihead(q)  # [B, H, 1, D/H]
    k_multi = kv_cache["keys"].transpose(1, 2)  # [B, H, T+1, D/H]

    print("q_multi shape:", q_multi.shape)
    print("k_multi shape:", k_multi.shape)

    
    # 6. 计算注意力分数（带因果掩码）
    attn_scores = torch.matmul(q_multi, k_multi.transpose(-2, -1)) / (d_model ** 0.5)

    print("attn_scores shape:", attn_scores.shape)

    # attn_scores = attn_scores.masked_fill(causal_mask[:step+1, :step+1], float('-inf'))
    
    # print("attn_scores shape:", attn_scores.shape)

    # 7. 注意力权重计算
    attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, 1, T+1]
    
    # 8. 加权求和
    output = torch.matmul(attn_weights, kv_cache["values"].transpose(1, 2))  # [B, H, 1, D/H]
    
    # 9. 合并多头输出
    output = output.contiguous().view(batch_size, 1, d_model)  # [B, 1, D]
    
    print(f"Step {step} 输出:", output.shape)