from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset
from swanlab.integration.transformers import SwanLabCallback
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from peft import LoraConfig, TaskType, get_peft_model
import deepspeed

DS_CONFIG = "ds_z2_offload_config.json"

from typing import Optional, List, Union
import sys


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

model = get_peft_model(model, config)

ds_reason = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh",cache_dir = './data/reasoning',split="train[:4500]")

def generate_conversation(examples):
    questions  = examples["Question"]
    cots = examples["Complex_CoT"]
    solutions = examples["Response"]
    conversations = []
    for question,cot,solution in zip(questions,cots, solutions):
        conversations.append([
            {"role" : "user",      "content" : question},
            {"role" : "assistant", "content" : f'<think>{cot}</think>{solution}'},
        ])
    return { "conversations": conversations, }

# 将转换后的推理数据集应用对话模板
reasoning_conversations = tokenizer.apply_chat_template(
    ds_reason.map(generate_conversation, batched = True)["conversations"],
    tokenize = False
)

ds_no_reason = load_dataset("BAAI/IndustryInstruction_Health-Medicine",cache_dir = './data/Medicine',split="train[:1500]")

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(ds_no_reason)

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
    project="Qwen3-8B-fintune",
    experiment_name="Qwen3-8B-combind",
    description="使用通义千问Qwen3-8B模型在FreedomIntelligence/medical-o1-reasoning-SFT和BAAI/IndustryInstruction_Health-Medicine数据集上微调。",
    config={
        "model": "Qwen/Qwen3-8B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(combined_dataset),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }
)


trainer = SFTTrainer(
    model = model,
    processing_class=tokenizer,

    train_dataset = combined_dataset,
    eval_dataset = None,  # 可以设置评估数据集
    callbacks=[swanlab_callback],
    args = SFTConfig(
        output_dir="./lora_model_combind",
        per_device_train_batch_size = 1,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 16,  # 使用梯度累积模拟更大批次大小
        warmup_steps = 5,  # 预热步数
        num_train_epochs = 7,  
        learning_rate = 2e-4,   # 学习率（长期训练可降至2e-5）
        logging_steps = 5,  # 日志记录间隔
        optim = "adamw_8bit",  # 优化器
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度类型
        seed = 3407,  # 随机种子
        report_to = "none",   # 可设置为"wandb"等进行实验追踪
        fp16=True,
        max_grad_norm=1.0,
        deepspeed=DS_CONFIG,
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


class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""  # 用于存储完整输出

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写方法捕获最终文本"""
        self.generated_text += text  # 累积输出
        super().on_finalized_text(text, stream_end=stream_end)  # 保持原样输出到终端

    def get_output(self) -> str:
        """获取完整生成内容"""
        return self.generated_text.strip()

def ask(question, is_thinking=True, save_to_file=None):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=is_thinking
    )

    # 使用自定义的 CaptureStreamer
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    # 生成响应
    model.eval()  # 确保模型在推理模式
    with torch.no_grad():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,  
            top_k=20,
            streamer=streamer,  # 关键：使用自定义的 streamer
        )

    # 获取完整输出
    full_output = streamer.get_output()

    # 保存到文件
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"✅ 成功写入文件: {save_to_file}")
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

    return full_output

# 测试调用
# ask("问：月经不调经色发黑而且量少。2012年起到现在一直月经很少，而且经色发黑，每个月来2天就没有了，吃过很多调月经的药，也在当地和县城里多次治疗过，可都没用，2014年1月13日，做了个人流后月经越是少还是黑的。有过多次打胎",
#     save_to_file='./output.txt')
# print("#############################################################################################")
# print("#############################################################################################")
# print("#############################################################################################")
# print("#############################################################################################")
# print("#############################################################################################")


# ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",is_thinking=False)

model.save_pretrained("lora_model_combind")  # Local saving
tokenizer.save_pretrained("lora_model_combind")
