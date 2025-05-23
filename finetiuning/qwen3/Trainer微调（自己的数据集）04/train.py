from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset
from swanlab.integration.transformers import SwanLabCallback
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, Trainer,DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import deepspeed
DS_CONFIG = "ds_zero2_no_offload.json"
from typing import Optional, List, Union
import sys
import json

model_name = "/root/autodl-tmp/Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

# 获取LoRA模型
# 转换模型
peft_model = get_peft_model(model, config)
peft_model.config.use_cache = False



def process_func_single(example, tokenizer):
    """
    医学问答数据集预处理（修复tokenizer未传递问题）
    Args:
        example: 包含question, cot, answer, type的样本
        tokenizer: 分词器对象（需支持apply_chat_template）
    Returns:
        包含input_ids, attention_mask, labels的字典
    """
    try:
        # 构建对话结构
        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": example["question"]}
        ]
        
        # 生成指令文本
        instruction_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        instruction = tokenizer(instruction_text, add_special_tokens=False)
        
        # 生成响应内容
        response_text = f"<think>{example['cot']}</think>\n{example['answer']}" if example["type"] == "reason" \
                       else f"<think>\n\n</think>\n{example['answer']}"
        
        response = tokenizer(response_text, add_special_tokens=False)
        
        # 合并结果
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    except Exception as e:
        print(f"处理样本时出错: {e}\n样本内容: {example}")
        raise



def process_func_batch(examples, tokenizer, max_length=2048):
    """
    批次处理函数（不填充到相同长度，但会截断超长部分）
    Args:
        examples: 包含question, cot, answer, type的批次数据
        tokenizer: 分词器对象（需支持apply_chat_template）
        max_length: 最大序列长度
    Returns:
        包含input_ids, attention_mask, labels的字典（各样本长度可能不同）
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for question, cot, answer, type in zip(examples["question"], examples["cot"], examples["answer"], examples["type"]):
        # 构建对话结构
        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": question}
        ]
        
        try:
            # 生成指令部分
            instruction_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            instruction = tokenizer(instruction_text, add_special_tokens=False)
            
            # 生成响应部分
            response_text = f"<think>{cot}</think>\n{answer}" if type == "reason" else f"<think>\n\n</think>\n{answer}"
            response = tokenizer(response_text, add_special_tokens=False)
            
            # 合并指令和响应
            input_ids = instruction["input_ids"] + response["input_ids"]+[tokenizer.eos_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"]+[1]
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]+[tokenizer.eos_token_id]
            
            # 截断超长部分
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
        except Exception as e:
            print(f"处理样本时出错 - 问题: {question[:50]}... 错误: {str(e)}")
            # 添加空样本以防中断流程
            batch_input_ids.append([])
            batch_attention_mask.append([])
            batch_labels.append([])
    
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }


# 处理数据集：读取json文件
# 创建 Dataset 对象

def load_json_to_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 加载为Python列表
    
    # 转换为Dataset对象
    return Dataset.from_list(data)

medical_dataset = load_json_to_dataset("combined_medical_data.json")


# train_dataset = medical_dataset.map(
#     lambda x: process_func_single(x, tokenizer),
#     batched=False
# )

train_dataset = medical_dataset.map(
    lambda x: process_func_batch(x, tokenizer),
    batched=True,
    batch_size=4
)




# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-fintune",
    experiment_name="Qwen3-8B-combind-trainer",
    description="使用通义千问Qwen3-8B模型在FreedomIntelligence/medical-o1-reasoning-SFT和BAAI/IndustryInstruction_Health-Medicine数据集上微调。",
    config={
        "model": "Qwen/Qwen3-8B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(train_dataset),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
)


# 配置训练参数
args = TrainingArguments(
    output_dir="./lora_train",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=4,
    save_steps=50,
    learning_rate=2e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    # bf16=True,
    fp16=True,
    max_grad_norm=1.0, 
    deepspeed=DS_CONFIG
)
        


# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


# 开启模型训练
trainer.train()
trainer.save_model('./lora_train')
trainer.save_state()
