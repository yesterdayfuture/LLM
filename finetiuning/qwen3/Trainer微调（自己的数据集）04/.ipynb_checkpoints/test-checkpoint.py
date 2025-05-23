from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import TextStreamer


# 原始模型路径和训练后的LoRA路径
base_model_path = "/root/autodl-tmp/Qwen/Qwen3-8B"  # 原始模型路径
lora_model_path = "./lora_train"                    # 你保存的LoRA模型路径

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

# 测试集中的数据
ask("一名31岁男性，左侧阴囊软性肿块持续存在2年，平时无不适，卧位不消失。结婚4年未育，查体发现左侧阴囊明显松弛下垂，似蚯蚓团块状物清晰可见，触之柔软，透光试验阴性，精液常规检查提示弱精症。根据这些症状和检查结果，他最可能的诊断是什么疾病？")

print("\n\n\n")
print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("\n\n\n")

ask("一名31岁男性，左侧阴囊软性肿块持续存在2年，平时无不适，卧位不消失。结婚4年未育，查体发现左侧阴囊明显松弛下垂，似蚯蚓团块状物清晰可见，触之柔软，透光试验阴性，精液常规检查提示弱精症。根据这些症状和检查结果，他最可能的诊断是什么疾病？",is_thinking=False)


# 训练集中的推理数据
# ask("对于以兴奋躁动为主要临床表现但同时患有肝功能障碍的精神分裂症病人，术前应该选用哪种药物？")

# print("\n\n\n")
# print("####################################################")
# print("####################################################")
# print("####################################################")
# print("####################################################")
# print("\n\n\n")

# ask("对于以兴奋躁动为主要临床表现但同时患有肝功能障碍的精神分裂症病人，术前应该选用哪种药物？",is_thinking=False)


# 训练集中的对话数据
# ask("我口腔内出现了一些白色的斑块，表面粗糙，颗粒状，有些部位还出现了红色颗粒，大约持续了一个星期左右。同时在舌背和右颊内侧也有一些白色斑块。我没有其他明显的病史和疾病，请问我应该怎么办？")

# print("\n\n\n")
# print("####################################################")
# print("####################################################")
# print("####################################################")
# print("####################################################")
# print("\n\n\n")

# ask("我口腔内出现了一些白色的斑块，表面粗糙，颗粒状，有些部位还出现了红色颗粒，大约持续了一个星期左右。同时在舌背和右颊内侧也有一些白色斑块。我没有其他明显的病史和疾病，请问我应该怎么办？",is_thinking=False)


# 其他问题
# ask("1+1=？")

# print("\n\n\n")
# print("####################################################")
# print("####################################################")
# print("####################################################")
# print("####################################################")
# print("\n\n\n")

# ask("1+1=？",is_thinking=False)
