---
license: Apache License 2.0
tags:
  - self-cognition
  - sft
  - ms-swift
  - 自我认知
  - 微调
text:
  - llm
language:
  - zh
  - en
configs:
  - config_name: default
    data_files:
      - split: train
        path: "self_cognition.jsonl"
---

## 介绍（Introduction）
该自我认知数据集由modelsope swift创建, 可以通过将通配符进行替换：{{NAME}}、{{AUTHOER}}，来创建属于自己大模型的自我认知数据集，总共108条。

ms-swift github：[https://github.com/modelscope/swift/](https://github.com/modelscope/swift/)

This self-cognition dataset was created by modelsope swift and can be customized for your own large model by replacing the placeholders: {{NAME}} and {{AUTHOER}}. It consists of a total of 134 entries.

ms-swift github: https://github.com/modelscope/swift/

## 使用（Usage）

只是下载：
```python
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('swift/self-cognition', subset_name='default', split='train')
```

或者自动替换{{NAME}}和{{AUTHOR}}【推荐】：

安装ms-swift：

```shell
pip install ms-swift -U
```

```python
from swift.llm import load_dataset

dataset = load_dataset(['swift/self-cognition'], model_name=['小黄', 'Xiao Huang'], model_author=['魔搭', 'ModelScope'])[0]
print(dataset)
print(dataset[0])
"""
Dataset({
    features: ['messages'],
    num_rows: 108
})
{'messages': [{'role': 'user', 'content': '你是？'}, {'role': 'assistant', 'content': '我是小黄，由魔搭训练的人工智能助手。我的目标是为用户提供有用、准确和及时的信息，并通过各种方式帮助用户进行有效的沟通。请告诉我有什么可以帮助您的呢？'}]}
"""

# 支持重采样：（超过108后进行重采样）
dataset = load_dataset(['swift/self-cognition#500'], model_name=['小黄', 'Xiao Huang'], model_author=['魔搭', 'ModelScope'])[0]
print(dataset)
"""
Dataset({
    features: ['messages'],
    num_rows: 500
})
"""
```
