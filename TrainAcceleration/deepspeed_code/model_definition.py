#导入环境
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data():
    #下载数据集
    # 1. 定义数据转换（预处理）
    transform = transforms.Compose([
        transforms.ToTensor(),          # 转为Tensor格式（自动归一化到0-1）
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST的均值和标准差）
    ])
    
    # 2. 下载数据集
    train_data = datasets.MNIST(
        root='./data',          # 数据存储路径
        train=True,           # 训练集
        download=True,        # 自动下载
        transform=transform   # 应用预处理
    )
    
    test_data = datasets.MNIST(
        root='./data',
        train=False,          # 测试集
        transform=transform
    )


    # 3. 创建数据加载器（自动分批次）
    train_loader = DataLoader(train_data, batch_size=64, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=2, shuffle=False)

    return train_loader, test_loader


#定义模型结构
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层组合
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),   # 输入1通道，输出32通道，3x3卷积核
            nn.ReLU(),              # 激活函数
            nn.MaxPool2d(2),        # 最大池化（缩小一半尺寸）

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),           # 展平多维数据
            nn.Linear(64*5*5, 128), # 输入维度需要计算（后面解释）
            nn.ReLU(),
            nn.Linear(128, 10)      # 输出10个数字的概率
    )

    def forward(self, x):
        x = self.conv_layers(x)     # 通过卷积层
        x = self.fc_layers(x)       # 通过全连接层
        return x