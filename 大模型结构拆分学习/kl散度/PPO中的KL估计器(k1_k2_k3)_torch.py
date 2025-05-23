
import torch

#蒙特卡洛估计变种（k1/k2/k3）

def compute_kl_estimator(log_p, log_q, estimator_type="k3"):
    log_ratio = log_p - log_q  # 假设log_p和log_q已归一化
    
    if estimator_type == "k1":
        return log_ratio.mean()
    elif estimator_type == "k2":
        return (log_ratio**2 / 2).mean()
    elif estimator_type == "k3":
        return (torch.exp(log_ratio) - 1 - log_ratio).mean()
    else:
        raise ValueError("Unsupported estimator type")

# 使用示例
p_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
q_logits = torch.tensor([[2.0, 3.0, 1.0], [5.0, 4.0, 6.0]])
kl = compute_kl_estimator(p_logits, q_logits)
print(kl)



