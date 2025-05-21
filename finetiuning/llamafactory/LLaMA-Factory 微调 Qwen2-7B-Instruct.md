### 一、系统环境
- 使用的 autoDL 算力平台

#### 1、下载基座模型

```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com  # （可选）配置 hf 国内镜像站

huggingface-cli download --resume-download shenzhi-wang/Llama3-8B-Chinese-Chat --local-dir /root/autodl-tmp/models/Llama3-8B-Chinese-Chat1
```

### 二、llama factory 框架
#### 1、安装框架
```shell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

#### 2、准备数据




**将训练数据放在 LLaMA-Factory/data/fintech.json**
**并且修改数据注册文件：LLaMA-Factory/data/dataset_info.json**

```powershell
"fintech": {
  "file_name": "fintech.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "history": "history"
  }
}
```

#### 3、启动 webui 界面
- 注意：使用下述命令 将远程端口 转发到 本地端口

```shell
ssh -CNg -L 7860:127.0.0.1:7860 -p 12610 root@connect.nmb2.seetacloud.com
```

-  webui 启动命令

```shell
cd LLaMA-Factory
llamafactory-cli webui
```

- 启动成功显示
- ![image-20250514210143455](/Users/zhangtian/Library/Application Support/typora-user-images/image-20250514210143455.png)
### 四、在 webui 中设置相关参数

### 五、进行微调
#### 1、方式一：在 webui 界面上进行微调
**前提：已完成 第四步**

#### 2、方式二：根据 第四步 生成的参数，使用命令行进行微调
**前提：已完成 第四步**

### 六、微调前后（聊天结果）进行对比

- 微调前

- 微调后
