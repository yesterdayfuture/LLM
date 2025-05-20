
'''
使用命令行进行启动

启动命令如下：
deepspeed ds_train.py --epochs 10 --deepspeed --deepspeed_config ds_config.json
'''

import argparse
import torch
import torchvision
import deepspeed
from model_definition import load_data, CustomModel



if __name__ == '__main__':
    #读取命令行 传递的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", help = "local device id on current node", type = int, default=0)
    parser.add_argument("--epochs", type = int, default=1)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    #获取数据集
    train_loader, test_loader = load_data() #数据集加载器中的 batch_size的大小 = （ds_config.json中 train_batch_size/gpu数量）

    #获取原始模型
    model = CustomModel().cuda()

    #将模型 包装成 deepspeed 形式
    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss().cuda() # 损失函数（分类任务常用）
    
    for i in range(args.epochs):
        for inputs, labels in train_loader:
            #前向传播
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model_engine(inputs)
            loss = loss_fn(outputs, labels)

            #使用 deepspeed 进行 反向传播和梯度更新
            #反向传播
            model_engine.backward(loss)
        
            #梯度更新
            model_engine.step()
        model_engine.save_checkpoint('./ds_models', i)

    #模型保存
    torch.save(model_engine.module.state_dict(),'deepspeed_train_model.pth')
