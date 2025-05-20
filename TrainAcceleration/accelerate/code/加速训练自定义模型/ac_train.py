'''
在终端使用命令行执行此文件
命令如下:
accelerate launch ac_train.py

'''

#导入环境
from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_definition import load_data, CustomModel


if __name__ == '__main__':

    #加载数据
    train_loader, test_loader = load_data()
    #定义模型
    model = CustomModel()

    #定义 accelerate 实例，并使用 deepspeed 插件
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

    #定义 优化器、损失函数
    optimization = torch.optim.Adam(model.parameters(), lr=0.00015)
    crition = torch.nn.CrossEntropyLoss()

    #将 模型、数据集、优化器 进行包装
    model, train_loader, optimization = accelerator.prepare(model, train_loader, optimization)

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = crition(outputs, labels)
            
            optimization.zero_grad()
            accelerator.backward(loss)
            optimization.step()
        print(f"Epoch {epoch} loss: {loss.item()}")
            
    accelerator.save(model.state_dict(), "./model_save/model.pth")

    





