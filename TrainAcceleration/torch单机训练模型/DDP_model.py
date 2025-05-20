'''
DDP训练启动命令

# 单机双卡启动命令
torchrun \
    --nproc_per_node=2 \  # 使用2个GPU
    --nnodes=1 \          # 单机模式
    --node_rank=0 \       # 当前机器rank
    --master_addr="127.0.0.1" \
    --master_port=12345 \
    ddp_mnist.py（文件名）
'''


#导入环境
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

#导入 DDP训练 需要的环境
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch.utils.data.distributed import DistributedSampler
import argparse #读取命令行参数
from torch.nn.parallel import DistributedDataParallel as DDP

def load_data(world_size,local_rank):
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

    # 2.1 配置分布式采样器（确保不同GPU获取不同数据子集）
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=local_rank)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank)
    # 3. 创建数据加载器（自动分批次）
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, sampler = train_sampler)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, sampler = test_sampler)

    return train_loader, test_loader, train_sampler, test_sampler


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

if __name__=="__main__":
    #设置参数args.local_rank
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", help = "local device id on current node", type = int)
    # args = parser.parse_args()


    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_size = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        # 检查GPU是否可用，以及当前 GPU 数量
        print("当前设备:", device)
        print("GPU数量：", world_size)
    else:
        print("GPU 不能使用")
        raise Exception("GPU 不能使用")


    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank) # 绑定当前进程到指定GPU

    #加载数据集
    train_loader, test_loader, train_sampler, test_sampler = load_data(world_size, local_rank)

    #模型载入args.local_rank
    # 创建模型实例
    device = torch.device(f"cuda:{local_rank}")
    model = CustomModel().to(device)

    #原始模型 包装 为 DDP
    DDP_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    #定义 损失函数，优化器，训练次数
    loss_fn = nn.CrossEntropyLoss()          # 损失函数（分类任务常用）
    optimize = optim.Adam(model.parameters(), lr=0.001)  # 优化器（自动调节学习率）
    
    train_step=0
    test_step=0
    epoch=10


    #开始训练
    #开始训练时间
    start = time.perf_counter()
    
    for i in range(epoch):
        print("------第 {} 轮训练开始------".format(i+1))
 
        train_sampler.set_epoch(epoch) #每张卡在每个周期上的值是随机的，设置epoch保证shuffle有效性
 
        DDP_model.train()
        for batch in train_loader:
            inputs,labels = batch
 
            #将数据赋值到args.local_rank
            inputs = inputs.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            #记录 单批数据 训练时间
            starttime = time.time()
            
            outputs = DDP_model(inputs)
            
            loss = loss_fn(outputs, labels)
            
            optimize.zero_grad()
            
            loss.backward()
            
            optimize.step()
            
            endtime = time.time()
    
            train_step=train_step+1
            if train_step%100==0:
                print("训练次数:{},Loss:{},time:{}".format(train_step,loss.item(),endtime-starttime))
 
        #仅在alocal_rank == 0时保存
        if local_rank ==0:
            torch.save(DDP_model,"./DDPtrain/DDP_model_{}.pth".format(i))
        print("模型已保存")

    end = time.perf_counter()

    #训练时间
    print(f"DDP训练使用时间: {end - start} 秒")
    
    # 清理分布式环境
    torch.distributed.destroy_process_group()

    



