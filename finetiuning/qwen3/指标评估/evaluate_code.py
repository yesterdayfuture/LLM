import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import pandas as pd
import re
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# 配置参数
MODEL_NAME = "/root/autodl-tmp/Qwen/Qwen3-8B"
LORA_PATH = "./lora_train"
TEST_DATA_PATH = "combined_medical_test.json"
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 4
NUM_SAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# 初始化语义评估模型
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 忽略警告
warnings.filterwarnings("ignore")

# 加载模型和tokenizer
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )
    # model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    # 确保所有模型在相同设备上
    device = model.device
    semantic_model.to(device)
    
    return tokenizer, model, device

tokenizer, model, device = load_models()

# 数据加载和预处理
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:NUM_SAMPLES]

test_data = load_test_data(TEST_DATA_PATH)

# 改进的文本清理函数（处理思考模式输出的<think>标签）
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 移除思考模式标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>|<\|endoftext\|>', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 计算指标（返回整数百分比）
def calculate_token_metrics(preds, refs):
    preds = [p for p in preds if p]
    refs = [r for r in refs if r]
    
    if not preds or not refs or len(preds) != len(refs):
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    
    precisions, recalls, f1s, accuracies = [], [], [], []
    
    for pred, ref in zip(preds, refs):
        pred_set = set(pred.split())
        ref_set = set(ref.split())
        common = pred_set & ref_set
        precision = 100 * len(common) / len(pred_set) if pred_set else 0
        recall = 100 * len(common) / len(ref_set) if ref_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = 100 * len(common) / len(ref_set) if ref_set else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
    
    return {
        "precision": int(np.round(np.mean(precisions))),
        "recall": int(np.round(np.mean(recalls))),
        "f1": int(np.round(np.mean(f1s))),
        "accuracy": int(np.round(np.mean(accuracies)))
    }

def calculate_semantic_metrics(preds, refs):
    preds = [p if p else " " for p in preds]
    refs = [r if r else " " for r in refs]
    
    if not preds or not refs or len(preds) != len(refs):
        return {
            "semantic_cosine_mean": 0,
            "semantic_cosine_std": 0,
            "semantic_cosine_min": 0,
            "semantic_cosine_max": 0
        }
    
    batch_size = 32
    cos_sims = []
    for i in range(0, len(preds), batch_size):
        batch_preds = preds[i:i+batch_size]
        batch_refs = refs[i:i+batch_size]
        with torch.no_grad():
            pred_embs = semantic_model.encode(batch_preds, convert_to_tensor=True)
            ref_embs = semantic_model.encode(batch_refs, convert_to_tensor=True)
            batch_cos_sims = torch.nn.functional.cosine_similarity(pred_embs, ref_embs)
            cos_sims.extend(batch_cos_sims.cpu().tolist())
    
    cos_sims = [100 * x for x in cos_sims]  # 转换为百分比
    return {
        "semantic_cosine_mean": int(np.round(np.mean(cos_sims))),
        "semantic_cosine_std": int(np.round(np.std(cos_sims))),
        "semantic_cosine_min": int(np.round(np.min(cos_sims))),
        "semantic_cosine_max": int(np.round(np.max(cos_sims)))
    }

def calculate_classic_metrics(preds, refs):
    preds = [p if p else " " for p in preds]
    refs = [r if r else " " for r in refs]
    
    if not preds or not refs or len(preds) != len(refs):
        return {
            "rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "bleu": 0,
            "precision": 0, "recall": 0, "f1": 0, "accuracy": 0
        }
    
    # ROUGE (转换为百分比)
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(preds, refs)
        rouge_metrics = {
            "rouge-1": int(round(100 * sum(s['rouge-1']['f'] for s in rouge_scores) / len(rouge_scores))),
            "rouge-2": int(round(100 * sum(s['rouge-2']['f'] for s in rouge_scores) / len(rouge_scores))),
            "rouge-l": int(round(100 * sum(s['rouge-l']['f'] for s in rouge_scores) / len(rouge_scores)))
        }
    except:
        rouge_metrics = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    
    # BLEU (转换为百分比)
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for p, r in zip(preds, refs):
        try:
            bleu_scores.append(100 * sentence_bleu([r.split()], p.split(), smoothing_function=smoothie))
        except:
            bleu_scores.append(0)
    rouge_metrics["bleu"] = int(round(sum(bleu_scores) / len(bleu_scores)))
    
    # Token级指标
    token_metrics = calculate_token_metrics(preds, refs)
    rouge_metrics.update(token_metrics)
    
    return rouge_metrics

def batch_generate(questions, question_types):
    all_responses = []
    
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Generating"):
        batch = questions[i:i+BATCH_SIZE]
        batch_types = question_types[i:i+BATCH_SIZE]
        
        # 为每个问题准备输入（根据type决定enable_thinking）
        inputs = []
        for q, q_type in zip(batch, batch_types):
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": q}
            ]
            inputs.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=(q_type == "reason")  # 关键修改：根据type启用思考模式
                )
            )
        
        # Tokenize并生成
        inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        device = next(model.parameters()).device
        # 将所有张量移动到指定的设备上
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码并清理
        decoded = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=False
        )
        cleaned = [clean_text(d) for d in decoded]
        all_responses.extend(cleaned)
        
        # 清理内存
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return all_responses

def evaluate():
    questions = [d["question"] for d in test_data]
    references = [d["answer"] for d in test_data]
    cots = [d.get("cot", "") for d in test_data]
    question_types = [d["type"] for d in test_data]
    
    # 生成回答
    answers = batch_generate(questions, question_types)
    
    # 准备评估数据
    eval_data = []
    for i, (q, ref, cot, q_type) in enumerate(zip(questions, references, cots, question_types)):
        eval_data.append({
            "question": q,
            "generated": answers[i] if i < len(answers) else "",
            "reference": ref,
            "cot_reference": cot,
            "type": q_type
        })
    
    # 计算指标
    metrics = {}
    
    # 1. Answer部分评估（所有样本）
    gen_answers = [d["generated"] for d in eval_data]
    ref_answers = [d["reference"] for d in eval_data]
    
    metrics["answer"] = {
        "classic": calculate_classic_metrics(gen_answers, ref_answers),
        "semantic": calculate_semantic_metrics(gen_answers, ref_answers)
    }
    
    # 2. CoT部分评估（仅reason类型且有cot_reference的样本）
    cot_data = [d for d in eval_data if d["type"] == "reason" and d["cot_reference"]]
    if cot_data:
        gen_cots = [d["generated"] for d in cot_data]
        ref_cots = [d["cot_reference"] for d in cot_data]
        
        metrics["cot"] = {
            "classic": calculate_classic_metrics(gen_cots, ref_cots),
            "semantic": calculate_semantic_metrics(gen_cots, ref_cots)
        }
        print(f"\nEvaluated {len(cot_data)} CoT samples")
    else:
        print("\nNo CoT samples to evaluate")
        metrics["cot"] = None
    
    # 3. Combined评估（所有样本，reason类型拼接cot）
    combined_gen = []
    combined_ref = []
    for d in eval_data:
        if d["type"] == "reason" and d["cot_reference"]:
            # 对于有CoT的样本，拼接生成内容和参考CoT
            combined_gen.append(f"{d['generated']} {d['cot_reference']}")
            combined_ref.append(f"{d['reference']} {d['cot_reference']}")
        else:
            # 对于没有CoT的样本，只使用回答部分
            combined_gen.append(d["generated"])
            combined_ref.append(d["reference"])
    
    metrics["combined"] = {
        "classic": calculate_classic_metrics(combined_gen, combined_ref),
        "semantic": calculate_semantic_metrics(combined_gen, combined_ref)
    }
    
    # 保存结果
    pd.DataFrame(eval_data).to_csv("base_results.csv", index=False)
    with open("base_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics, eval_data

def print_metrics(name, data):
    if data is None:
        print(f"\n[{name.upper()}] No data to display")
        return
    
    print(f"\n[{name.upper()}]")
    print("Classic Metrics:")
    print(f"  ROUGE-1: {data['classic']['rouge-1']}%")
    print(f"  ROUGE-2: {data['classic']['rouge-2']}%")
    print(f"  ROUGE-L: {data['classic']['rouge-l']}%")
    print(f"  BLEU:    {data['classic']['bleu']}%")
    print(f"  Precision: {data['classic']['precision']}%")
    print(f"  Recall:    {data['classic']['recall']}%")
    print(f"  F1:        {data['classic']['f1']}%")
    print(f"  Accuracy:  {data['classic']['accuracy']}%")
    print("\nSemantic Metrics:")
    print(f"  Cosine Mean: {data['semantic']['semantic_cosine_mean']}%")
    print(f"  Cosine Std:  ±{data['semantic']['semantic_cosine_std']}%")
    print(f"  Cosine Min:  {data['semantic']['semantic_cosine_min']}%")
    print(f"  Cosine Max:  {data['semantic']['semantic_cosine_max']}%")

if __name__ == "__main__":
    metrics, results = evaluate()
    
    print("\n=== Evaluation Results ===")
    print(f"\nEvaluated {len(results)} samples")
    
    print_metrics("Answer", metrics["answer"])
    if "cot" in metrics and metrics["cot"] is not None:
        print_metrics("Chain-of-Thought", metrics["cot"])
    print_metrics("Combined", metrics["combined"])