from transformers import AutoModelForVision2Seq, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel,get_peft_model_state_dict
import deepspeed
import torch
import os
from datasets import load_from_disk, features
DS_CONFIG = "ds_z2_offload_config.json"


device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device_map,
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct", do_image_splitting=False,use_fast=True)


#数据集从 本地 加载

dataset = load_from_disk("/root/finetune_vl/LLaMA-Factory/llamafactory/RLHF-V2")

#数据格式转换
def data_cov(example):
    
    prompt = [{
        'role':'user',
        'content':[
            {"type": "image"}, 
            {"type": "text", "text": example["conversations"][0]['value'].replace("<image>","")}
        ]
    }]

    chosen = [{
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]['value']}],
        }]
    

    rejected = [{
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]['value']}],
        }]

    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)

    return {
        'prompt':prompt,
        'chosen':chosen,
        'rejected':rejected,
        'images':[example["images"][0]]
    }

process_data = dataset['train'].select(range(1000)).map(data_cov, remove_columns=['conversations'])
# f = process_data.features
# f["images"] = features.Sequence(features.Image(decode=True)) # to avoid bytes
# process_data = process_data.cast(f)



# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=32,  # Lora 秩_set_static_graph
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
# 转换模型
peft_model = get_peft_model(model, config)
peft_model.config.use_cache = False
peft_model._set_gradient_checkpointing()



# Train the model
training_args = DPOConfig(
    output_dir="./output2/Qwen2.5-VL-DPO",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    # use_reentrant=False,
    logging_steps=5,
    save_steps=100,
    learning_rate=1e-4,
    logging_first_step=5,
    deepspeed=DS_CONFIG,
    max_grad_norm=1.0,
    beta=0.8,
    report_to='swanlab'
)

# class CustomDPOTrainer(DPOTrainer):
#     def tokenize_row(self, feature):
#         # 提取多模态数据
#         prompt = feature["prompt"]
#         images = feature["images"]
        
#         # 显式传递文本和图像参数 [3](@ref)
#         processed_inputs = self.processor(
#             text=prompt,  # 强制命名参数
#             images=images, 
#             add_special_tokens=False,
#             return_tensors="pt"
#         )
#         return processed_inputs

# # 初始化 Trainer
# trainer = CustomDPOTrainer(
#     peft_model,
#     args=training_args,
#     train_dataset=process_data,
#     tokenizer=processor,
#     ref_model=None
# )


trainer = DPOTrainer(
    peft_model,
    ref_model=None, # not needed when using peft
    args=training_args,
    train_dataset=process_data,
    processing_class=processor,
    
)

trainer.train()


#模型合并
merged_model= peft_model.merge_and_unload()

#模型保存
merged_model.save_pretrained("/root/autodl-tmp/merged_model")

#tokenizer合并
processor.save_pretrained("/root/autodl-tmp/merged_model")






