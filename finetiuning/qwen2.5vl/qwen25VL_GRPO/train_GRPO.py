import os 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from datasets import load_dataset,Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info
from trl import GRPOConfig, GRPOTrainer
from transformers import (
AutoModelForCausalLM, 
AutoTokenizer, 
Qwen2_5_VLForConditionalGeneration, 
AutoProcessor,
BitsAndBytesConfig
)
import torch
import deepspeed
DS_CONFIG = "ds_z2_offload_config.json"
import json
from datasets import load_dataset,Dataset
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from trl import GRPOConfig
from grpo_trainer import Qwen2VLGRPOTrainer # third-party trainer from open-R1

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  

tokenizer = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct",
                                         use_fast=True)
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels,use_fast=True)
# use cuda device
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct", 
                                             device_map=device_map, 
                                            torch_dtype=compute_dtype,
                                            quantization_config=bnb_config
                                                          )
 
#ds = load_dataset("BUAADreamer/llava-med-zh-instruct-60k",split = "train[0:2000]",cache_dir='./data').select(range(100))
ds = load_from_disk("/root/autodl-tmp/qwen25VL_GRPO/GRPO-V2")
 
 
def get_prompt_rft(example):
    '''
    input: dict example, including PIL image object
    output: multiple samples, within a dict format
    '''
    dialogue_num = len(example['messages'])
    i = 0
    results=[]
    while i<dialogue_num:
        assert example['messages'][i]['role']=='user' and example['messages'][i+1]['role']=='assistant'
        question_sample = example['messages'][i]['content']
        answer_sample = example['messages'][i+1]['content']
        img_pil = example['images'][0].resize((112,112))  # reduce vRAM burden
        out_results = []
        SYSTEM_PROMPT = r'''
        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.
        Before answering, think carefully about the question and create a step-by-step chain of 
        thoughts to ensure a logical and accurate response.
        
        ### Instruction:
        You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
        Please answer the following medical question based on the input image. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output format should be as follows:
<think> ... </think> <answer>...</answer>
除了特殊符号，请用中文回答
        '''   # for a different language, please change the last few words.
        results.append({
                'prompt': [
                    {'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {'role': 'user', 'content': [
                        {"type": "image", },  
                        {"type": "text", "text": question_sample},    
                    ]}
                ],
                'image':img_pil,
                'solution':answer_sample,
            })
        i+=2
    return results
 
def dataset_gen():
    for items in ds:
        multiple_out = get_prompt_rft(items)
        for single_out in multiple_out:
            yield single_out
my_gen = dataset_gen()
 
dataset_train = Dataset.from_generator(dataset_gen)





import re

from Levenshtein import ratio as levenshtein_ratio
def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # print(completions) #debug
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matches = [re.match(pattern, content[0]['content'], re.DOTALL) for content in completions]
    for content in completions:
        print('prediction=='+content[0]['content']+'\n\n')
    return [1.0 if match else 0.0 for match in matches]
def levenshtein_reward_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    print(res)
    print('\n\n')
    return res


output_dir="./outputs/Qwevl-Instruct-GRPO"
run_name="Qwen-vl-GRPO-medical"

model.train()
peft_config = LoraConfig(
    r=8, #Rank
    lora_alpha=16,
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        # "gate_proj", 
        # "up_proj", 
        # "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
)
 
training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = False,
    fp16 = True,
    per_device_train_batch_size = 1,# keep same with num_generations
    gradient_accumulation_steps = 16, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 2048,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 100,
    save_steps = 5,

    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    deepspeed=DS_CONFIG,
    disable_tqdm=False,  # 确保不禁用进度条
)
trainer = Qwen2VLGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func, # all reward functions
        levenshtein_reward_func],
    args=training_args,
    train_dataset=dataset_train,
    peft_config = peft_config,
)
 
trainer.train()
 
trainer.save_model(output_dir)