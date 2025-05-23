from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import TextStreamer


# 原始模型路径和训练后的LoRA路径
base_model_path = "/root/autodl-tmp/Qwen/Qwen3-8B"  # 原始模型路径
lora_model_path = "./lora_model_no_reason"                    # 你保存的LoRA模型路径

# ---------- 1. 加载分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(base_model_path)  # 直接加载你保存的分词器

# ---------- 2. 加载基础模型 ----------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自动分配设备（GPU/CPU）
)

# ---------- 3. 加载LoRA适配器 ----------
model = PeftModel.from_pretrained(base_model, lora_model_path)

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
            temperature = 0.6, top_p = 0.95, top_k = 20, # 推理模式参数
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
ask("问：月经不调经色发黑而且量少。2012年起到现在一直月经很少，而且经色发黑，每个月来2天就没有了，吃过很多调月经的药，也在当地和县城里多次治疗过，可都没用，2014年1月13日，做了个人流后月经越是少还是黑的。有过多次打胎")


print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("\n\n\n")

ask("问：月经不调经色发黑而且量少。2012年起到现在一直月经很少，而且经色发黑，每个月来2天就没有了，吃过很多调月经的药，也在当地和县城里多次治疗过，可都没用，2014年1月13日，做了个人流后月经越是少还是黑的。有过多次打胎",is_thinking=False)
