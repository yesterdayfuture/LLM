

import torch
import torch.nn.functional as F


# KL散度计算
def kl_divergence(p_logits, q_logits):

    q = F.softmax(q_logits, dim=-1)
    print(f"q: {q}")

    log_p = F.log_softmax(p_logits, dim=-1)
    print(f"log_p: {log_p}")

    log_q = F.log_softmax(q_logits, dim=-1)
    print(f"log_q: {log_q}")

    return (q * (log_q - log_p)).sum(dim=-1).mean()

# 使用示例
p_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
q_logits = torch.tensor([[2.0, 3.0, 1.0], [5.0, 4.0, 6.0]])
kl = kl_divergence(p_logits, q_logits)
print(kl)