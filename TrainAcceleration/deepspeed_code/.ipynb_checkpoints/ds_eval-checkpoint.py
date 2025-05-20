
import argparse
import torch
import torchvision
import deepspeed
from model_definition import load_data, CustomModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义数据转换（预处理）
transform = transforms.Compose([
    transforms.ToTensor(),          # 转为Tensor格式（自动归一化到0-1）
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST的均值和标准差）
])

test_data = datasets.MNIST(
        root='./data',
        train=False,          # 测试集
        transform=transform
    )

#获取数据集
train_loader, test_loader = load_data()

model = CustomModel()
model.load_state_dict(torch.load('deepspeed_train_model.pth'))

#评估
model.eval()  # 设置为评估模式
correct = 0
total = 0

with torch.no_grad():  # 不计算梯度（节省内存）
    for images, labels in test_loader:
        images, labels = images, labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")


# 随机选择一张测试图片
index = np.random.randint(0,1000)  # 可以修改这个数字试不同图片
test_image, true_label = test_data[index]
test_image = test_image.unsqueeze(0)  # 增加批次维度

# 预测
with torch.no_grad():
    output = model(test_image)
predicted_label = torch.argmax(output).item()

print(f"预测: {predicted_label}, 真实: {true_label}")

# 显示结果
plt.imshow(test_image.cpu().squeeze(), cmap='gray')
plt.title(f"预测: {predicted_label}, 真实: {true_label}")
plt.show()
