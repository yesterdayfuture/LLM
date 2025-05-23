

import torch
import torch.nn.functional as F

# Jensen-Shannon 散度
def js_divergence(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    print(f"p: {p}")

    q = F.softmax(q_logits, dim=-1)
    print(f"q: {q}")

    m = 0.5 * (p + q)
    print(f"m: {m}")
    
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()

# 使用示例
p_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
q_logits = torch.tensor([[2.0, 3.0, 1.0], [5.0, 4.0, 6.0]])
kl = js_divergence(p_logits, q_logits)
print(kl)

