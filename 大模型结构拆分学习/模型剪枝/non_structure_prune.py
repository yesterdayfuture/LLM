'''
自定义实现非结构化剪枝

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.nn import GELU
from torch.nn.functional import gelu


# 定义全连接网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Resize((28, 28))])
train_dataset = datasets.MNIST('/Users/zhangtian/work/llm_study/mnistdata', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('/Users/zhangtian/work/llm_study/mnistdata', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)


#训练原始模型
def train(model, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # print(data.shape)cl

        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} | Loss: {loss.item():.4f}')

# 定义模型
model = nn.Sequential(
        nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
)
print("模型结构：", model)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 评估模型 函数
def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f'测试准确率: {accuracy:.2f}%')



# 模型训练，训练3个epoch
# for epoch in range(1, 4):
#     train(model, optimizer, epoch)

#模型评估
evaluate(model)

'''
自定义剪枝函数
'''

# 打印 模型参数
print("模型参数：", model.named_parameters())
for name, param in model.named_parameters():
    print(name, param.size())
print("模型各层详细解释：", model.named_modules())
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        for name1, param in module.named_parameters():
            if 'weight' in name1:
                # print(param.data)
                #打印 当前层数， 当前层数的参数名，参数形状，参数量(param.data.numpy().size 或 param.numel())
                print(name, name1, param.data.numpy().shape, param.data.numpy().size, param.numel())


'''
torch.quantile 用于计算输入张量在指定维度上的分位数。
分位数表示数据分布中的特定位置，例如中位数（0.5 分位数）、四分位数（0.25 和 0.75 分位数）等。
​​适用场景​​：
    模型剪枝（确定权重阈值）
    数据分布分析（如识别异常值）
    分位数回归（损失函数设计）
'''
# 定义手动剪枝函数，amount表示剪枝比例
def manual_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for name1, param in module.named_parameters():
                if 'weight' in name1:# 仅处理全连接层的权重
                    weights = param.data.abs()         # 取绝对值作为重要性度量
                    threshold = torch.quantile(weights.flatten(), amount)  # 计算剪枝阈值
                    mask = (weights > threshold).float()  # 生成掩码（保留高于阈值的权重）
                    param.data *= mask                     # 应用掩码（剪枝）

# 执行剪枝
manual_pruning(model, amount=0.3)
print("-------------剪枝完成-------------")

# 统计稀疏性
def print_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for name1, param in module.named_parameters():
                if 'weight' in name1:# 仅处理全连接层的权重
                    sparsity = 100 * (param == 0).sum().item() / param.numel()
                    print(f"{name} 稀疏度: {sparsity:.2f}%")

print_sparsity(model)


# # 微调时需冻结被剪枝的权重（设为非可训练）
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         param.requires_grad = False  # 冻结权重
#         param.data[param == 0] = 0  # 强制置零（避免梯度更新恢复剪枝）

# # 微调2个epoch
# for epoch in range(1, 3):
#     train(model, optimizer, epoch)

# #模型评估
# evaluate(model)


