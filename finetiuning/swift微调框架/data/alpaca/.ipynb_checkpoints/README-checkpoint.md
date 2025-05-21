---
license: CC BY NC 4.0
#用户自定义标签
tags:
- finetune
- alpaca
- gpt4

text:
  TextGeneration:
    样本规模:
      - 10k-100k
    language:
      - zh
    语言:
      - 中文
---


## 数据集描述
该数据集为GPT-4生成的中文数据集，用于LLM的指令精调和强化学习等。



### 数据集加载方式
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("alpaca-gpt4-data-zh", namespace="AI-ModelScope", split="train")
print(next(iter(ds)))
```

### 数据分片
数据已经预设了train分片。



## 数据集版权信息
数据集已经开源，license为CC BY NC 4.0（仅用于非商业化用途），如有违反相关条款，随时联系modelscope删除。


## 引用方式
```
@article{peng2023gpt4llm,
    title={Instruction Tuning with GPT-4},
    author={Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao},
    journal={arXiv preprint arXiv:2304.03277},
    year={2023}
}
```

## 参考链接
```
https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh
https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
```

### Clone with HTTP
```bash
git clone https://www.modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh.git
```