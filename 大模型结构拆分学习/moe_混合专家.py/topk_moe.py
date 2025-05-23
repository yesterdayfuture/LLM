
import torch
import torch.nn as nn

# 定义专家网络
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)


# 定义混合专家网络
class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, top_k):
        super().__init__()
        #输入维度
        self.input_dim = input_dim
        #输出维度
        self.output_dim = output_dim
        #专家数量
        self.num_experts = num_experts
        #获取前几位专家
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        # 门控计算
        weights = torch.softmax(self.gate(x), dim=-1).reshape(-1, self.num_experts)  # [batch_size, num_experts]
        # print(weights.shape)
        # print(f"weights: {weights}")

        #获取前 k 位专家
        top_k_weights,top_k_indics = torch.topk(weights,self.top_k,dim=-1)
        print(top_k_weights.shape)
        print(f"top_k_weights: {top_k_weights}")
        print(top_k_indics.shape)
        print(f"top_k_indics: {top_k_indics}")

        # 专家输出加权
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2).reshape(-1, self.num_experts, self.output_dim)  # [batch, seq, experts, dim]
        # print(expert_outputs.shape)
        # print(f"expert_outputs: {expert_outputs}")


        i,j,_ = x.shape

        outputs = torch.zeros((i*j, self.output_dim))

        #i*j 表示 batch_size*seq_len，即所有 token 数
        for k in range(i*j):
            for l in top_k_indics[k]:
                outputs[k] += expert_outputs[k,l]*weights[k,l]
        return outputs.reshape(i,j,-1)

# 示例
moe = MoE(num_experts=4, input_dim=512, output_dim=512, top_k=2)
x = torch.randn(2, 10, 512)
print(moe(x).shape)  # torch.Size([2, 10, 512])


