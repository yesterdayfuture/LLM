


### 一、前提条件
#### 1、系统环境（使用的 autodl 算力平台）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eface659ddab477db87577dd6904b2bf.png)
#### 2、安装相关库

##### 安装 modescope

```shell
pip3 install -U modelscope
# 或使用下方命令
# pip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

##### 安装 fastchat

```shell
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install .
```

##### 使用 魔搭 安装模型

```python

from modelscope.hub.snapshot_download import snapshot_download

local_dir_root = "/root/autodl-tmp"
snapshot_download('baichuan-inc/Baichuan2-13B-Chat', cache_dir=local_dir_root)

```

### 二、使用fastchat
一共要启动三个服务分别是controller、model_worker（vllm 使用vllm_worker）、openai_api_server

 - vllm 加快推理速度：就是快点给出问题的答案（当前vllm可用，可不用）

```shell
#安装vllm
pip install vllm
```
#### 1、第一步启动controller

```shell
python -m fastchat.serve.controller --host 0.0.0.0

```
- 运行结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/25d680f0c89642d4ae8cfad9f6caf246.png)



 - --host参数指定应用程序绑定的主机名或IP地址。默认情况下，应用程序将绑定在本地回环地址（即localhost或127.0.0.1）上。
   --port参数指定应用程序监听的端口号。默认情况下，应用程序将监听21001端口。
   --dispatch-method参数指定请求调度算法。lottery表示抽奖式随机分配请求，shortest_queue表示将请求分配给队列最短的服务器。默认情况下，使用抽奖式随机分配请求。
   --ssl参数指示应用程序是否使用SSL加密协议。如果指定了此参数，则应用程序将使用HTTPS协议。否则，应用程序将使用HTTP协议。

#### 2、启动model_worker（llm）

 - 安装accelerate

```shell
pip install accelerate
```
- 启动model_worker

```shell
python -m fastchat.serve.model_worker --model-path /root/autodl-tmp/models_from_modelscope/baichuan-inc/Baichuan2-13B-Chat --host 0.0.0.0  --load-8bit

```

- 运行结果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7deae62ab131405d82421320fdcf318f.png)


 - 相关参数
 

```python
--host：指定模型工作进程绑定的主机名或IP地址。

--port：指定模型工作进程绑定的端口号。

--worker-address：指定模型工作进程的地址。

--controller-address：指定模型控制器的地址。

--model-path：指定要加载的模型文件的路径。

--revision：指定模型文件的版本。

--device：指定模型运行的设备类型，可以是CPU、GPU等。

--gpus：指定用于模型运行的GPU设备的数量。

--num-gpus：指定用于模型运行的GPU设备的数量。

--max-gpu-memory：指定用于模型运行的GPU设备的最大内存限制。

--dtype：指定模型的数据类型，可以是float32、float16等。

--load-8bit：启用8位量化模型。

--cpu-offloading：启用CPU卸载。

--gptq-ckpt：指定GPTQ检查点的路径。

--gptq-wbits：指定GPTQ权重的位数。

--gptq-groupsize：指定GPTQ分组大小。

--awq-ckpt：指定AWQ检查点的路径。

--awq-wbits：指定AWQ权重的位数。

--awq-groupsize：指定AWQ分组大小。

--enable-exllama：启用Exllama。

--exllama-max-seq-len：指定Exllama的最大序列长度。

--exllama-gpu-split：指定Exllama的GPU划分。

--exllama-cache-8bit：启用Exllama的8位缓存。

--enable-xft：启用XFT。

--xft-max-seq-len：指定XFT的最大序列长度。

--xft-dtype：指定XFT的数据类型。

--model-names：指定要加载的模型文件的名称。

--conv-template：指定卷积模板的路径。

--embed-in-truncate：启用嵌入截断。

--limit-worker-concurrency：限制工作进程并发性的数量。

--stream-interval：指定流间隔。

--no-register：不注册模型。

--seed：指定随机种子。

--debug：启用调试模式。

--ssl：启用SSL。

```

#### 2、**第二步代替方案(vllm)**

- 运行指令为
```shell
python -m fastchat.serve.vllm_worker --model-path /root/autodl-tmp/models_from_modelscope/baichuan-inc/Baichuan2-13B-Chat  --host 0.0.0.0  
```
- 相关参数

```python
--host HOST：指定该工作节点的主机名或 IP 地址，默认为 localhost。
--port PORT：指定该工作节点监听的端口号，默认为 8000。
--worker-address WORKER_ADDRESS：指定该工作节点的地址。如果未指定，则自动从网络配置中获取。
--controller-address CONTROLLER_ADDRESS：指定控制节点的地址。如果未指定，则自动从环境变量中获取。如果环境变量也未设置，则默认使用 http://localhost:8001。
--model-path MODEL_PATH：指定模型文件的路径。如果未指定，则默认使用 models/model.ckpt。
--model-names MODEL_NAMES：指定要加载的模型名称。该参数只在多模型情况下才需要使用。
--limit-worker-concurrency LIMIT_WORKER_CONCURRENCY：指定最大并发工作进程数。默认为 None，表示不限制。
--no-register：禁止在控制节点上注册该工作节点。
--num-gpus NUM_GPUS：指定使用的 GPU 数量。默认为 1。
--conv-template CONV_TEMPLATE：指定对话生成的模板文件路径。如果未指定，则默认使用 conversation_template.json。
--trust_remote_code：启用远程代码信任模式。
--gpu_memory_utilization GPU_MEMORY_UTILIZATION：指定 GPU 内存使用率，范围为 [0,1]。默认为 1.0，表示占用全部 GPU 内存。
--model MODEL：指定要加载的模型类型。默认为 fastchat.serve.vllm_worker.VLLMModel。
--tokenizer TOKENIZER：指定要使用的分词器类型。默认为 huggingface。
--revision REVISION：指定加载的模型版本号。默认为 None，表示加载最新版本。
--tokenizer-revision TOKENIZER_REVISION：指定加载的分词器版本号。默认为 None，表示加载最新版本。
--tokenizer-mode {auto,slow}：指定分词器模式。默认为 auto，表示自动选择最佳模式。
--download-dir DOWNLOAD_DIR：指定模型下载目录。默认为 downloads/。
--load-format {auto,pt,safetensors,npcache,dummy}：指定模型加载格式。默认为 auto，表示自动选择最佳格式。
--dtype {auto,half,float16,bfloat16,float,float32}：指定模型数据类型。默认为 auto，表示自动选择最佳类型。
--max-model-len MAX_MODEL_LEN：指定模型的最大长度。默认为 None，表示不限制。
--worker-use-ray：启用 Ray 分布式训练模式。
--pipeline-parallel-size PIPELINE_PARALLEL_SIZE：指定管道并行的大小。默认为 None，表示不使用管道并行。
--tensor-parallel-size TENSOR_PARALLEL_SIZE：指定张量并行的大小。默认为 None，表示不使用张量并行。
--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS：指定最大并发加载工作数。默认为 4。
--block-size {8,16,32}：指定块大小。默认为 16。
--seed SEED：指定随机种子。默认为 None。
--swap-space SWAP_SPACE：指定交换空间的大小。默认为 4GB。
--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS：指定每个批次的最大令牌数。默认为 2048。
--max-num-seqs MAX_NUM_SEQS：指定每个批次的最大序列数。默认为 64。
--max-paddings MAX_PADDINGS：指定每个批次的最大填充数。默认为 1024。
--disable-log-stats：禁止记录统计信息。
--quantization {awq,gptq,squeezellm,None}：指定模型量化类型。默认为 None，表示不进行量化。
--enforce-eager：强制启用 Eager Execution 模式。
--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE：指定要捕获的上下文长度。默认为 1024。
--engine-use-ray：在引擎中启用 Ray 分布式训练模式。
--disable-log-requests：禁止记录请求信息。
--max-log-len MAX_LOG_LEN：指定最大日志长度。默认为 10240。

```

#### 3、第三步openai服务启动
- 需要重新起一个 **终端**

```shell
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000
```

- 运行结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a039d77c6ed7478f8a1c106e869236aa.png)

### 验证
- 安装验证环境

```shell
pip install langchain
pip install openai
```
- 代码

```python
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    streaming=True,
    verbose=True,
    # callbacks=[callback],
    openai_api_key="none",
    openai_api_base="http://localhost:8000/v1",
    model_name="Baichuan2-13B-Chat"
)



# 提示词
template = """
我很想去{location}旅行，我应该在哪里做什么？
"""
prompt = PromptTemplate(
    input_variables=["location"],
    template=template,

)
# 说白了就是在提示词的基础上，把输入的话进行格式化方法输入，前后添加了一些固定词
final_prompt = prompt.format(location='安徽合肥')

print(f"最终提升次：{final_prompt}")
output = llm([HumanMessage(content=final_prompt)])
print(f"LLM输出结果：{output}")

```

- 运行结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0aa4ec3226244ecea3d377c0ee1379fe.png)

### fastchat 可使用浏览器启动页面（运行前 先完成上述所有流程）
- 安装环境

```shell
pip install gradio
```
- 启动页面

```shell
python3 -m fastchat.serve.gradio_web_server

```

- 运行结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bee62abef67f4ab0b14a66dd60173d68.png)
- 运行界面

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aabdb8c93b6143389074ff541bf25b4e.png)

#### 注意
- 本地主机 对 远程服务器 进行 端口转发

```shell
ssh -CNg -L 7860:127.0.0.1:7860  -p 32946 root@connect.nmb2.seetacloud.com
```
最后输入 远程服务器 密码
