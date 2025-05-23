
import torch

#Rényi散度（α=0.5）​

def renyi_divergence(p_logits, q_logits, alpha=0.5):

    p = torch.nn.functional.softmax(p_logits, dim=-1)

    q = torch.nn.functional.softmax(q_logits, dim=-1)

    sum_term = (p**alpha * q**(1-alpha)).sum(dim=-1)
    
    return (1 / (alpha - 1)) * torch.log(sum_term).mean()

# 使用示例
p_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
q_logits = torch.tensor([[2.0, 3.0, 1.0], [5.0, 4.0, 6.0]])
kl = renyi_divergence(p_logits, q_logits)
print(kl)

