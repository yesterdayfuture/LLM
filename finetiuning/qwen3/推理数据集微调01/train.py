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

#加载模型时，直接加载在多卡上（deepspeed和accelerate是先将模型加载在单卡上，运行时转移到多卡上，缺点是：一旦模型过大，加载容易失败）
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)


'''
enable_input_require_grads() 功能解析
‌一、核心作用‌
‌强制输入张量计算梯度‌

默认情况下PyTorch不计算输入张量的梯度，调用此方法后会修改模型前向传播逻辑，
使inputs_embeds等输入参数自动获得requires_grad=True属性
主要用于‌参数高效微调(PEFT)‌场景，如LoRA微调时需要计算输入层梯度
‌
典型应用场景‌

结合梯度检查点(gradient checkpointing)技术时，需确保输入参与梯度计算
修改模型输入嵌入(如inputs_embeds)时的梯度传播需求
'''

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

#将 lora模块 与 原始模型 关联
model = get_peft_model(model, config)

ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh",cache_dir = './dir',split="train[:5000]")

#将数据格式转换为 对话组（AIpaca 格式）
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


# 将转换后的推理数据集应用对话模板（chatLM格式）
reasoning_conversations = tokenizer.apply_chat_template(
    ds.map(generate_conversation, batched = True)["conversations"],
    tokenize = False
)

#打乱数据集
df = pd.DataFrame({"text": reasoning_conversations})
train_ds = Dataset.from_pandas(df).shuffle(seed = 3407)


#记录训练过程中 的 损失，需要注册 SwanLab 网站的账号
swanlab_callback = SwanLabCallback(
    project="Qwen3-4B-fintune",
    experiment_name="Qwen3-4B",
    description="使用通义千问Qwen3-4B模型在FreedomIntelligence/medical-o1-reasoning-SFT数据集上微调。",
    config={
        "model": "Qwen/Qwen3-4B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(train_ds),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }
)

#配置 参数高效微调训练器 参数
trainer = SFTTrainer(
    model = model,
    processing_class=tokenizer,
    train_dataset = train_ds,
    eval_dataset = None,  # 可以设置评估数据集
    callbacks=[swanlab_callback],
    args = SFTConfig(
        output_dir="./lora_model",
        per_device_train_batch_size = 1,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 16,  # 使用梯度累积模拟更大批次大小
        warmup_steps = 5,  # 预热步数
        num_train_epochs = 4,  # 训练轮数设置为1以进行完整训练
        learning_rate = 2e-4,   # 学习率（长期训练可降至2e-5）
        logging_steps = 5,  # 日志记录间隔
        optim = "adamw_8bit",  # 优化器
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度类型
        seed = 3407,  # 随机种子
        report_to = "none",   # 可设置为"wandb"等进行实验追踪
        fp16=True,
        max_grad_norm=1.0,
        deepspeed=DS_CONFIG, # deepspeed 加速
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

#开始训练
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
        add_generation_prompt=True
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
ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",
    save_to_file='./output.txt')
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")


# ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",is_thinking=False)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
