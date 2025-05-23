from unsloth import FastLanguageModel
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset,load_from_disk,DatasetDict
from swanlab.integration.transformers import SwanLabCallback
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from peft import LoraConfig, TaskType, get_peft_model

from typing import Optional, List, Union
import sys
import deepspeed

DS_CONFIG = "ds_z2_offload_config.json"

model_name = "/root/autodl-tmp/Qwen/Qwen3-4B"


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,   
        
    full_finetuning = False,    
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,           #  LoRA秩，建议值为8,16,32,64,128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # LoRA alpha值，建议设为rank或rank*2
    lora_dropout = 0.1, # LoRA dropout，0值经过优化
    bias = "none",    # 偏置设置，"none"已优化
    
    # [新特性] "unsloth"模式减少30%显存，可适应2倍大的批次大小
    use_gradient_checkpointing = "unsloth", #梯度检查点，用于长上下文
    random_state = 3407,  # 随机种子
    use_rslora = False,   # 是否使用rank stabilized LoRA
    loftq_config = None,  # LoftQ配置
)

#数据集从 本地 加载

ds_reason = load_from_disk("./local_reason_dataset")
ds_reason = DatasetDict({
    'train': Dataset.from_dict(ds_reason['train'][:1600]) 
})

def convert_to_alpaca_format(examples):
    conversations = []
    for messages in examples["messages"]:
        # 提取系统提示词
        system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        
        # 提取用户问题和助手回答
        user_question = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        assistant_answer = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        
        # 分割think和response部分
        think_part = ""
        response_part = ""
        if assistant_answer:
            think_start = assistant_answer.find("<think>")
            think_end = assistant_answer.find("</think>")
            if think_start != -1 and think_end != -1:
                think_part = assistant_answer[think_start+7:think_end].strip()
            
            response_start = assistant_answer.find("<response>")
            response_end = assistant_answer.find("</response>")
            if response_start != -1 and response_end != -1:
                response_part = assistant_answer[response_start+10:response_end].strip()
            else:
                # 如果没有response标签，尝试获取think标签之后的内容
                response_part = assistant_answer[think_end+8:].strip()
        
        # 构建Alpaca格式的对话
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f'<think>{think_part}</think>{response_part}'}
        ]
        conversations.append(conversation)
    
    return {"conversations": conversations}

# 批量处理数据集
ds_reason_alpaca = ds_reason['train'].map(
    convert_to_alpaca_format,
    batched=True,
    remove_columns=ds_reason['train'].column_names,
)

# 将转换后的推理数据集应用对话模板
reasoning_conversations = tokenizer.apply_chat_template(
    ds_reason_alpaca["conversations"],
    tokenize = False
)


ds_no_reason = load_from_disk("./local_no_reason_dataset")
ds_no_reason = DatasetDict({
    'train': Dataset.from_dict(ds_no_reason['train'][:400]) 
})

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(ds_no_reason['train'])

# 将标准化后的非推理数据集应用对话模板
non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)

# 合并两个数据集
data = pd.concat([
    pd.Series(reasoning_conversations),    # 推理对话数据
    pd.Series(non_reasoning_conversations)        # 采样后的非推理对话数据
])

data.name = "text"  # 设置数据列名为"text"

# 将合并的数据转换为HuggingFace Dataset格式
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
# 随机打乱数据集
combined_dataset = combined_dataset.shuffle(seed = 3407)



swanlab_callback = SwanLabCallback(
    project="Qwen3-4B-fintune",
    experiment_name="Qwen3-4B-combind",
    description="使用通义千问Qwen3-4B模型在FreedomIntelligence/medical-o1-reasoning-SFT和BAAI/IndustryInstruction_Health-Medicine数据集上微调。",
    config={
        "model": "Qwen/Qwen3-4B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(combined_dataset),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }
)


trainer = SFTTrainer(
    model = model,
    tokenizer=tokenizer,
    dataset_text_field = "text",
    train_dataset = combined_dataset,
    eval_dataset = None,  # 可以设置评估数据集
    callbacks=[swanlab_callback],
    args = SFTConfig(
        output_dir="./lora_model",
        per_device_train_batch_size = 1,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 4,  # 使用梯度累积模拟更大批次大小
        warmup_steps = 5,  # 预热步数
        num_train_epochs = 3,  
        learning_rate = 2e-4,   # 学习率（长期训练可降至2e-5）
        logging_steps = 5,  # 日志记录间隔
        optim = "adamw_8bit",  # 优化器
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度类型
        seed = 3407,  # 随机种子
        report_to = "none",   # 可设置为"wandb"等进行实验追踪
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        # deepspeed=DS_CONFIG,
        logging_first_step=5,
        save_steps=100,
    ),
)

# 显示当前内存统计
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 显示最终内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
