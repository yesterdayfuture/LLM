import torch.nn as nn
import torch

# 示例：基于选择概率的负载均衡损失（PyTorch实现）  
class LoadBalanceLoss(nn.Module):  
    def __init__(self, num_experts):  
        super().__init__()  
        self.num_experts = num_experts  

    def forward(self, gates):  
        # 计算各专家的平均负载  
        expert_load = torch.mean(gates, dim=1)  
        print(f"expert_load: {expert_load}")

        # 计算负载均衡损失（标准差最小化）  
        balance_loss = torch.std(expert_load)  
        return balance_loss  

a = torch.Tensor([[1,2,3],[4,5,6]])

lossObj = LoadBalanceLoss(3)
print(lossObj(a))


'''
使用PyTorch从零实现MoE模型并集成辅助负载均衡损失的完整方案
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义专家模型
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.net(x)


#带噪声的门控网络
class NoisyGating(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.w_gate = nn.Linear(input_dim, num_experts)
        self.w_noise = nn.Linear(input_dim, num_experts)
        self.k = k
        
    def forward(self, x):
        # 计算门控分数和噪声
        logits = self.w_gate(x)  # [B, seq_len, num_experts]
        noise = torch.randn_like(logits) * F.softplus(self.w_noise(x))
        
        # 添加噪声并选择Top-K
        noisy_logits = logits + noise
        topk_vals, topk_idx = torch.topk(noisy_logits, self.k, dim=-1)
        
        # 生成稀疏掩码
        mask = torch.zeros_like(noisy_logits).scatter_(-1, topk_idx, 1)
        weights = F.softmax(topk_vals, dim=-1)
        
        return weights, topk_idx, mask
    

#重要性损失（Importance Loss）
def load_balancing_loss(gate_probs, expert_mask, num_experts):
    """
    gate_probs: 门控概率 [B, seq_len, num_experts]
    expert_mask: 专家选择掩码 [B, seq_len, num_experts]
    """
    # 计算每个专家的使用频率
    expert_count = expert_mask.sum(dim=(0,1))  # [num_experts]

    # 计算总token数 或 计算所有专家 总的使用频率
    total_tokens = expert_mask.sum()

    # 计算专家使用频率的均值
    expert_util = expert_count / total_tokens
    
    # 计算门控概率的均值
    gate_mean = gate_probs.mean(dim=(0,1))  # [num_experts]
    
    # 负载均衡损失Switch Transformers
    '''
    乘以专家数量的作用：解决专家利用不均衡问题
    乘以专家数量 N 的作用‌：
        ‌归一化尺度‌：当专家数量增加时，损失值随 N 线性增长，避免损失因专家数量不同而出现量级不平衡，确保训练稳定性45。
        ‌强化均衡信号‌：通过放大损失梯度，促使模型更关注专家利用率的均衡性，防止某些专家被过度激活或完全闲置
    '''
    load_loss = torch.sum(expert_util * gate_mean) * num_experts

    # 负载均衡损失 basic
    load_loss2 = torch.sum(expert_util * gate_mean)

    return load_loss

#专家容量惩罚（Expert Capacity Penalty）
def expert_capacity_penalty(expert_counts, capacity_factor=1.0):
    """
    expert_counts: 每个专家的token计数 [num_experts]
    capacity = (total_tokens / num_experts) * capacity_factor
    """
    capacity = (expert_counts.sum() / len(expert_counts)) * capacity_factor
    over_limit = F.relu(expert_counts - capacity)
    return torch.mean(over_limit)


#MoE模型
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=8, hidden_dim=512, k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = NoisyGating(input_dim, num_experts, k)
        self.num_experts = num_experts
        self.k = k
        
    def forward(self, x):
        B, seq_len, d_model = x.shape
        weights, topk_idx, mask = self.gate(x)  # [B, seq_len, k]

        print("weights", weights.shape)
        # print(weights)
        print("topk_idx", topk_idx.shape)
        # print(topk_idx)
        print("mask", mask.shape)
        # print(mask)

        
        # 专家输出计算
        expert_input = x.view(-1, d_model)  # [B*seq_len, d_model]
        expert_outputs = []
        
        for i in range(self.k):
            # 为每个选中的专家分配输入
            expert_idx = topk_idx[..., i].view(-1)  # [B*seq_len]
            print("expert_idx", expert_idx.shape)
            print(expert_idx)

            expert_out = torch.zeros_like(expert_input)
            
            # 并行计算所有专家（需处理未选中情况）
            for expert_id in range(self.num_experts):
                idx_mask = (expert_idx == expert_id)
                if idx_mask.any():
                    expert_out[idx_mask] = self.experts[expert_id](expert_input[idx_mask])
            
            expert_outputs.append(expert_out.view(B, seq_len, d_model))
        
        # 加权合并输出
        combined = sum(w.unsqueeze(-1) * out for w, out in zip(weights.unbind(-1), expert_outputs))
        
        print("combined", combined.shape)

        # 计算负载损失
        gate_probs = F.softmax(self.gate.w_gate(x), dim=-1)
        load_loss = load_balancing_loss(gate_probs, mask, self.num_experts)
        
        return combined, load_loss
    


'''
训练
'''
def train(dataloader, vocab_size):
    model = MoE(input_dim=768, num_experts=8, hidden_dim=2048)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for batch in dataloader:
        x, y = batch
        outputs, load_loss = model(x)
        
        # 主任务损失（例如语言建模）
        main_loss = F.cross_entropy(outputs.view(-1, vocab_size), y.view(-1))
        
        # 总损失（负载损失权重设为0.01[9](@ref)）
        total_loss = main_loss + 0.01 * load_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


a = torch.randn(2, 10, 32)
ceshi_model = MoE(input_dim=32, num_experts=8, hidden_dim=32)
outputs, loss = ceshi_model(a)
print(outputs.shape, loss)