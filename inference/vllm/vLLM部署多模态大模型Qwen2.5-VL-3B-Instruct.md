
### 一、前提环境
#### 1、系统环境
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c2718828ec5e4be980cb1018aa1b68e5.png)

#### 2、安装相关环境
- 安装依赖

```shell
%pip install accelerate qwen-vl-utils[decord]==0.0.8
%pip install transformers==4.50.0
%pip install modelscope==1.24.0
%pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
%pip install pillow requests python-dotenv
%pip install vllm==0.7.3

```
- 注意：flash-attn 不是必须安装的，可安可不安
- 注意：如果安装 flash-attn 失败，使用离线安装
下载地址：[flash-attn下载](https://github.com/Dao-AILab/flash-attention/releases)
- 下载界面：根据自己系统的版本进行下载
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6885f6c1e6ae4744a68dbe01167fd6c3.png)
- 安装命令

```shell
pip install flash-att本地地址
```

#### 3、查看相关环境

```shell
%pip show torchvision modelscope flash-attn
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/034a65d7d6ba40a7aed64d8081decbce.png)


```shell
%pip show transformers accelerate qwen-vl-utils
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a705b8901f4c47168f270e5450a3ed2f.png)



```shell
%pip show torch 
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc9687b00d844e48b117c69a8a1a61b4.png)


### 二、模型下载
- 使用 魔搭 下载模型

```python
# model_download.py
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='/root/autodl-tmp', revision='master')

```

### 三、运行模型
#### 1、方式一（使用代码 直接运行 模型）
- 首先从 vLLM 库中导入 LLM 和 SamplingParams 类。LLM 类是使用 vLLM 引擎运行离线推理的主要类。SamplingParams 类指定采样过程的参数，用于控制和调整生成文本的随机性和多样性。
- vLLM 提供了非常方便的封装，我们直接传入模型名称或模型路径即可，不必手动初始化模型和分词器
- 详细代码如下：

```python
# vllm_model.py
# 使用 vllm 本地模式 使用
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


class QwenVLModel:
    def __init__(self, model_path="/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_path = model_path

        self.llm = LLM(
            model=self.model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
            tensor_parallel_size=1,      # 设置为1以减少GPU内存使用
            gpu_memory_utilization=0.9,  # 控制GPU内存使用率
            max_model_len=2048,          # 限制最大序列长度
            # quantization="awq",        # 使用AWQ量化来减少内存使用
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=512,
            stop_token_ids=[],
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, messages):
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text



from tqdm import tqdm
img_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"  
prompt_str = "请用中文描述一下这张图片"
image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": prompt_str},
        ],
    },
]
model = QwenVLModel()
output_text = model.generate(image_messages)
print(output_text)

```

- 代码运行结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6cb5e0186441400aa0207ad97e0409dc.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c55766b4669c4e078c076c56fd969f56.png)



#### 2、方式二（提高 openai 式接口）
- 运行指令

```shell
#vllm 执行命令

python -m vllm.entrypoints.openai.api_server \
--model /root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct \
--served-model-name qwen-vl \
--max-model-len 64000 \
--limit-mm-per-prompt "image=5" \  # 允许每个prompt处理5张图像[9](@ref)
--port 8000
```
- 解释：

```
--port 参数指定地址。
–model 参数指定模型名称。
–served-model-name 指定服务模型的名称。
–max-model-len 指定模型的最大长度。
--limit-mm-per-prompt "image=5"  允许每个prompt处理5张图像
```

- 指令运行结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae10a6471ada47eaa53e99e75ddf28ad.png)





- 测试代码

```python
#使用langchain 调用 openai 的方式调用
# 引入 OpenAI 支持库  
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64, os

base_url ="http://localhost:8000/v1"  
api_key ="EMPTY"  

# 初始化LangChain客户端
llm = ChatOpenAI(
    model="qwen-vl",  # 与--served-model-name一致
    temperature=0.7,
    max_tokens=1024,
    base_url=base_url,  
    api_key=api_key
)

# 处理多模态输入（示例：文本+图像）
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 构造多模态消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请用中文描述这张图片的内容"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64('demo.jpeg')}"
                }
            }
        ]
    }
]

# 发送请求
response = llm.invoke(messages)
print(f"模型回复：{response.content}")
```

- 运行结果

```python
模型回复：这张图片展示了一位年轻女子和她的金毛犬在海滩上。女子坐在沙滩上，微笑着与狗狗互动。狗狗戴着项圈，看起来非常温顺和友好。背景是广阔的海洋，海浪轻轻拍打着海岸，整个场景充满了温馨和幸福的氛围。阳光洒在沙滩和海洋上，给人一种温暖而宁静的感觉。
```

#### 3、方式三（使用 transformers 运行模型）

```python
# 本地加载使用
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


class QwenVLModel:
    def __init__(self, model_path="/root/autodl-tmp/Qwen/Qwen2.5-VL-3B-Instruct", use_flash_attention=False):
        """
        初始化Qwen VL模型
        Args:
            model_path: 模型路径
            use_flash_attention: 是否使用flash attention加速
        """
        # 加载模型
        if use_flash_attention:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            )

        # 初始化处理器
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels, 
            use_fast=True
        )

    def process_image(self, image_path, prompt):
        """
        处理图片并生成输出
        Args:
            image_path: 图片路径
            prompt: 提示文本
        Returns:
            生成的文本输出
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 准备推理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text




#测试
model = QwenVLModel()
img_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
output_text = model.process_image(
    img_path,
    "请用中文描述一下这张图片"
)
print(f"输出信息: {output_text}")

```

