**PyTorch中提供了单卡训练、DataParallel（DP）和DistributedDataParallel（DDP），下面是相关原理与实现代码。**

 - 代码下载链接：[git代码链接](https://github.com/yesterdayfuture/pytorch_learn/tree/main/torch%E5%8D%95%E6%9C%BA%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)


### **一、单卡训练**
**原理**
单卡训练是最基础的模型训练方式，使用单个GPU（或CPU）完成所有计算任务：  
1. 模型与数据移动：将模型和数据加载到指定设备（如`cuda:0`），通过`model.to(device)`和`data.to(device)`实现。  
2. 前向与反向传播：在单一设备上完成前向计算、损失计算和梯度反向传播。  
3. 参数更新：优化器根据梯度更新模型参数。

**优点**
• 简单易用：无需处理并行逻辑，适合快速验证和小规模模型。  

• 调试方便：无多设备同步问题，错误排查直观。


**缺点**
• 资源受限：显存和计算能力受单卡限制，无法处理大模型或大数据集。  

• 效率瓶颈：无法利用多卡加速训练。


**实现方式**
```python
import torch

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型和数据
model = MyModel().to(device)
data = data.to(device)

# 训练循环
optimizer = torch.optim.Adam(model.parameters())
for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### **二、DataParallel（DP）**
**原理**
DP通过单进程多线程实现数据并行：  
1. 模型复制：主GPU（默认`cuda:0`）复制模型到其他GPU。  
2. 数据分片：将输入数据均分到各GPU，并行计算前向传播。  
3. 梯度汇总：各GPU的梯度回传至主GPU，求平均后更新模型参数，再同步到其他GPU。

**优点**
• 代码简单：仅需一行`model = nn.DataParallel(model)`即可启用多卡。  

• 快速实验：适合小规模多卡训练场景。


**缺点**
• 显存不均衡：主GPU需存储完整模型和汇总梯度，显存占用高，易成瓶颈。  

• 效率低下：线程间通信依赖GIL锁，并行加速比低。  

• 不支持多机：仅限单机多卡，无法扩展至多机。


**实现方式**
```python
import torch.nn as nn

# 检查多卡并包装模型
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
model.to(device)

# 注意：保存时需提取主模型
torch.save(model.module.state_dict(), "model.pth")
```

---

### **三、DistributedDataParallel（DDP）**
**原理**
DDP基于多进程实现分布式数据并行：  
1. 进程初始化：每个GPU对应独立进程，通过`init_process_group`初始化通信后端（如NCCL）。  
2. 模型复制：各进程加载相同模型，DDP自动同步初始参数。  
3. 数据分片：使用`DistributedSampler`确保各进程读取不重叠的数据。  
4. 梯度同步：反向传播时，梯度通过`All-Reduce`操作在各进程间同步，无需主GPU汇总。

**优点**
• 显存均衡：各GPU独立计算梯度，显存占用均匀。  

• 高效并行：多进程无GIL限制，通信开销低，加速比接近线性。  

• 扩展性强：支持多机多卡，适用于超大规模训练。


**缺点**
• 配置复杂：需手动设置进程组、数据采样器和启动脚本。  

• 调试困难：多进程环境下错误日志分散，排查难度高。


**实现方式**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
```

---

### **四、对比与选择建议**
| 特性               | 单卡训练       | DataParallel（DP） | DDP                |
|------------------------|------------------|-----------------------|-----------------------|
| 适用场景           | 小模型/调试       | 快速实验/单机多卡       | 大规模训练/多机多卡     |
| 显存占用           | 单卡满载          | 主卡显存瓶颈           | 各卡均衡               |
| 并行效率           | 无加速            | 低（线程通信开销）      | 高（进程级并行）        |
| 实现复杂度          | 简单              | 中等（代码简单）        | 复杂（需配置进程组）     |
| 扩展性             | 无                | 仅限单机               | 支持多机               |

**选择建议**
• 单卡训练：适合快速验证或显存需求极低的场景。  

• DP：仅推荐在单机多卡且代码快速迁移时使用，避免主卡显存瓶颈。  

• DDP：生产环境首选，尤其适合千亿参数模型或多节点训练。

