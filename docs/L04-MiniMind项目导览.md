# L04 - MiniMind 项目导览

> **"千里之行，始于配环境"**

---

## 📌 本节目标

1. 理解 MiniMind 项目的设计哲学和定位
2. 掌握项目的目录结构和每个关键文件的作用
3. 了解模型版本和核心配置参数
4. 完成环境搭建并运行第一次推理
5. 理解 MiniMind 的技术特色和面试价值

---

## 📚 前置知识

- 阅读完 L01-L03，理解大语言模型基本原理、Transformer 架构和 PyTorch 基础
- 能在终端中执行基本的 shell 命令（cd、ls、pip、python）
- 有一台安装了 Python 3.9+ 的电脑（有 GPU 更好，没有也可以用 CPU 体验推理）

---

## 正文讲解

### 1. MiniMind 项目的目标和设计哲学

#### 1.1 为什么需要 MiniMind？

想象你想学修汽车引擎。你有两个选择：

- **选项 A**：去法拉利工厂，面对一台 800 马力的 V12 发动机，上万个零件——看都看不懂
- **选项 B**：去教学工坊，面对一台结构完整但大小只有 1/2700 的教学引擎——每个零件都能拆开看

MiniMind 就是那台**教学引擎**。

| 大模型（GPT-3/4） | MiniMind |
|-------------------|----------|
| 参数 1750 亿+ | 参数仅 6400 万 |
| 需要上千张 A100 GPU | 一张消费级 GPU 即可 |
| 训练花费上亿美元 | 训练花费几块钱电费 |
| 代码用各种优化框架封装 | **纯 PyTorch 原生实现** |
| 只能用，看不懂 | **每一行代码都能读懂** |

#### 1.2 核心设计哲学

> **"麻雀虽小，五脏俱全"**

MiniMind 的设计目标不是"做一个好用的聊天机器人"，而是"用最小的代价，走完大模型训练的每一步"。因此它：

1. **覆盖全链路**：预训练 → SFT → DPO/RLHF → 推理评估，一步不少
2. **不依赖黑箱**：不用 HuggingFace Trainer、DeepSpeed 等高层封装，核心训练代码全部手写
3. **代码极简**：模型定义只有几百行，一个下午就能通读
4. **结构标准**：架构和 LLaMA 一脉相承，学会了 MiniMind 就理解了主流大模型的设计

### 2. 项目目录结构详解

下面是 MiniMind 的目录结构，以及每个文件夹/文件的作用：

```
minimind/
├── model/                        # 🧠 模型结构代码
│   ├── model_minimind.py         # ⭐ 核心：模型定义（MiniMindConfig、Attention、FFN、Block、CausalLM）
│   ├── model_lora.py             # LoRA 微调相关的模型定义
│   ├── tokenizer.json            # 分词器的词表和规则
│   └── tokenizer_config.json     # 分词器配置
│
├── trainer/                      # 🏋️ 训练脚本
│   ├── train_pretrain.py         # ⭐ 预训练脚本（从零训练语言模型）
│   ├── train_full_sft.py         # 全参数 SFT（有监督微调）
│   ├── train_lora.py             # LoRA 微调
│   ├── train_dpo.py              # DPO 偏好对齐训练
│   ├── train_distillation.py     # 知识蒸馏
│   ├── train_ppo.py              # PPO 强化学习训练
│   ├── train_grpo.py             # GRPO 训练
│   ├── train_agent.py            # Agent 训练（工具调用）
│   ├── train_tokenizer.py        # 训练自定义分词器
│   ├── trainer_utils.py          # 训练公共工具函数
│   └── rollout_engine.py         # PPO/GRPO 的 rollout 引擎
│
├── dataset/                      # 📊 数据集相关
│   ├── lm_dataset.py             # ⭐ 数据集定义（预训练、SFT 等数据加载逻辑）
│   ├── dataset.md                # 数据集说明文档
│   └── __init__.py
│
├── scripts/                      # 🛠️ 工具脚本
│   ├── web_demo.py               # Streamlit Web 对话界面
│   ├── chat_api.py               # 简单的 API 服务
│   ├── serve_openai_api.py       # OpenAI 兼容 API 服务
│   ├── convert_model.py          # 模型格式转换（支持 GGUF 等）
│   └── eval_toolcall.py          # 工具调用评估
│
├── eval_llm.py                   # 🎯 推理评估脚本（命令行对话）
├── requirements.txt              # 📦 Python 依赖列表
├── README.md                     # 📖 项目说明（中文）
├── README_en.md                  # 📖 项目说明（英文）
├── LICENSE                       # 📄 MIT 开源许可
└── images/                       # 🖼️ README 中使用的图片
```

#### 重点文件详解

##### `model/model_minimind.py` — 项目之魂

这是整个项目最重要的文件，所有模型结构都在这里。它包含：

| 类名 | 作用 | 对应概念 |
|------|------|---------|
| `MiniMindConfig` | 模型配置 | 定义所有超参数（隐藏层大小、层数、头数等） |
| `RMSNorm` | 归一化层 | L02 中讲的 Layer Normalization（简化版） |
| `Attention` | 注意力机制 | L02 中讲的 Self-Attention + GQA + RoPE |
| `FeedForward` | 前馈网络 | L02 中讲的 SwiGLU FFN |
| `MoEGate` / `MOEFeedForward` | 专家混合 | MoE 版本的 FFN（高级话题） |
| `MiniMindBlock` | Transformer 块 | 一层完整的 Transformer（Pre-Norm + Attention + FFN + 残差） |
| `MiniMindModel` | 模型主体 | Embedding + N × Block + 最终 RMSNorm |
| `MiniMindForCausalLM` | 因果语言模型 | MiniMindModel + lm_head + 损失计算 + 生成 |

##### `trainer/train_pretrain.py` — 训练入口

预训练脚本包含：

- 命令行参数解析（batch_size、learning_rate、epochs 等）
- 模型和优化器初始化
- 数据加载器构建
- 训练循环（混合精度、梯度累积、梯度裁剪）
- 模型保存和日志记录
- DDP 多卡分布式训练支持

##### `eval_llm.py` — 快速体验

推理脚本，加载训练好的模型权重，在命令行中与模型对话：

```python
# 核心流程
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MiniMindForCausalLM.from_pretrained(model_path)
# 自回归生成
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    top_p=args.top_p,
    temperature=args.temperature,
)
```

### 3. 模型版本介绍

MiniMind 提供了多个模型版本，核心区别在于参数量和是否使用 MoE：

| 模型版本 | 参数量 | 架构特点 | 适合场景 |
|---------|--------|---------|---------|
| **minimind-3** | ~64M | 标准 Dense 架构 | ⭐ 学习入门首选 |
| **minimind-3-MoE** | ~198M | Mixture of Experts | 学习 MoE 架构 |

> **MoE 是什么？** Mixture of Experts（专家混合）是一种"条件计算"技术——模型有多个 FFN "专家"，但每次只激活其中一部分。这样参数总量大（198M），但实际每次计算量和 64M 差不多。MoE 是 GPT-4 等前沿大模型采用的技术。

对于初学者，**强烈建议从 minimind-3（64M）开始**。它结构最简单，最容易理解。

### 4. 核心配置参数表

以下是 minimind-3 的默认配置（`MiniMindConfig` 类）：

| 参数名 | 值 | 含义 | 类比 |
|--------|-----|------|------|
| `hidden_size` | 768 | 隐藏层维度（d_model） | 每个词的"内部表示"有多丰富 |
| `num_hidden_layers` | 8 | Transformer 层数 | 模型有多"深"，信息被处理多少遍 |
| `num_attention_heads` | 8 | Query 注意力头数 | 从多少个角度理解句子 |
| `num_key_value_heads` | 4 | Key/Value 注意力头数（GQA） | KV Cache 的共享程度 |
| `head_dim` | 96 | 每个头的维度 (768/8) | 每个"角度"看到多少信息 |
| `vocab_size` | 6400 | 词表大小 | 模型认识多少个不同的 token |
| `intermediate_size` | ~2048 | FFN 中间层维度 | FFN 的"思考空间"大小 |
| `max_position_embeddings` | 32768 | 最大序列长度 | 一次最多处理多少 token |
| `dropout` | 0.0 | Dropout 概率 | 训练时随机关闭神经元的比例 |
| `use_moe` | False | 是否使用 MoE | minimind-3 为 False |

#### 参数量怎么算？

大语言模型的参数主要分布在这几个部分：

```
1. Embedding 层
   vocab_size × hidden_size = 6400 × 768 = 4,915,200 (≈5M)

2. 每个 Transformer Block（共 8 个）:
   - Q 投影: hidden_size × (num_attention_heads × head_dim) = 768 × 768 = 589,824
   - K 投影: hidden_size × (num_key_value_heads × head_dim) = 768 × 384 = 294,912
   - V 投影: 768 × 384 = 294,912
   - O 投影: 768 × 768 = 589,824
   - FFN gate: 768 × 2048 = 1,572,864
   - FFN up:   768 × 2048 = 1,572,864
   - FFN down: 2048 × 768 = 1,572,864
   - RMSNorm × 2: 768 × 2 = 1,536
   
   每层总计 ≈ 6.5M
   8 层总计 ≈ 52M

3. 最终 RMSNorm: 768
4. lm_head: 与 Embedding 共享权重，不额外算

总计 ≈ 5M + 52M + ... ≈ 64M
```

> **面试技巧**：能手算模型参数量是一个加分项，说明你真正理解了模型结构。

### 5. 环境搭建步骤

#### 第 1 步：克隆项目

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

#### 第 2 步：创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n minimind python=3.10
conda activate minimind

# 或者使用 venv
python -m venv venv
source venv/bin/activate
```

#### 第 3 步：安装 PyTorch

PyTorch 需要根据你的 GPU 和 CUDA 版本单独安装。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应命令。

```bash
# 有 NVIDIA GPU (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 只有 CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# macOS Apple Silicon
pip install torch torchvision torchaudio
```

#### 第 4 步：安装项目依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 主要包含：

| 包名 | 作用 |
|------|------|
| `transformers` | HuggingFace 模型加载/保存框架 |
| `datasets` | 数据集加载工具 |
| `trl` | 用于 DPO/PPO 等训练 |
| `swanlab` | 训练日志可视化 |
| `streamlit` | Web Demo 界面 |

#### 第 5 步：检查 GPU

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
```

即使没有 GPU，你也可以用 CPU 运行推理（只是慢一些），以及阅读和理解所有代码。

### 6. 快速体验：下载预训练模型并运行推理

MiniMind 提供了预训练好的模型权重，可以直接下载体验。

#### 方法 1：通过 HuggingFace 下载（推荐）

```bash
# 安装 huggingface_hub（如果还没装的话）
pip install huggingface_hub

# 下载模型（会自动缓存到本地）
python eval_llm.py
```

`eval_llm.py` 默认会从 HuggingFace Hub 加载预训练模型权重。

#### 方法 2：手动下载

从项目 README 中的链接下载模型文件，放到指定目录。

#### 运行推理

```bash
python eval_llm.py
```

运行后你会看到一个交互式对话界面：

```
用户: 你好，请介绍一下你自己
MiniMind: 我是 MiniMind，一个小型的语言模型...

用户: 什么是人工智能？
MiniMind: 人工智能是...
```

虽然 MiniMind 的回答质量远不如 GPT-4（毕竟只有 64M 参数），但你可以清楚地看到一个完整的语言模型是如何工作的。

#### 运行 Web Demo

```bash
streamlit run scripts/web_demo.py
```

这会启动一个浏览器界面，提供更友好的对话体验。

### 7. 训练数据的获取方式

MiniMind 使用的训练数据以 JSONL 格式存储。

#### 预训练数据

预训练数据是纯文本（无标注），格式如下：

```jsonl
{"text": "北京是中华人民共和国的首都，位于华北平原的北部..."}
{"text": "深度学习是机器学习的一个分支，它使用多层神经网络..."}
```

数据文件默认路径为 `dataset/pretrain_t2t_mini.jsonl`（相对于 trainer 目录）。

#### SFT 数据

SFT（有监督微调）数据是对话格式：

```jsonl
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}]}
```

数据加载逻辑在 `dataset/lm_dataset.py` 中实现。

### 8. MiniMind 的技术特色

#### 8.1 纯 PyTorch 原生实现

这是 MiniMind **最大的学习价值**。

很多开源项目虽然也提供了模型代码，但训练部分高度依赖 HuggingFace Trainer、DeepSpeed、Megatron 等高层框架。你看到的代码可能只有几行配置，真正的训练逻辑藏在框架内部。

MiniMind 不同——它的训练循环是**手写的**：

```python
# MiniMind 的训练循环（trainer/train_pretrain.py）
for step, (input_ids, labels) in enumerate(loader):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)

    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / args.accumulation_steps

    scaler.scale(loss).backward()

    if step % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

每一行你都能看懂——这就是 MiniMind 的价值。

#### 8.2 与 LLaMA 架构一脉相承

MiniMind 的架构选择和 Meta 的 LLaMA 系列高度一致：

| 技术 | LLaMA | MiniMind | 说明 |
|------|-------|----------|------|
| Pre-RMSNorm | ✅ | ✅ | 先归一化再计算 |
| SwiGLU FFN | ✅ | ✅ | 门控激活函数 |
| RoPE 位置编码 | ✅ | ✅ | 旋转位置编码 |
| GQA | ✅ | ✅ | 分组查询注意力 |
| Weight Tying | ✅ | ✅ | Embedding 和 lm_head 共享权重 |

学会了 MiniMind，你就理解了 LLaMA 的核心架构。

#### 8.3 覆盖全训练链路

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 分词器训练 | `train_tokenizer.py` | 从零构建 BPE 分词器 |
| 预训练 (PT) | `train_pretrain.py` | 从零训练语言模型，学习语言知识 |
| 全参 SFT | `train_full_sft.py` | 用对话数据微调，学习对话格式 |
| LoRA 微调 | `train_lora.py` | 低秩适应，小参数量微调 |
| DPO 对齐 | `train_dpo.py` | 直接偏好优化，对齐人类偏好 |
| PPO/GRPO | `train_ppo.py` / `train_grpo.py` | 强化学习方法对齐 |
| 知识蒸馏 | `train_distillation.py` | 从大模型学习知识 |
| Agent 训练 | `train_agent.py` | 工具调用能力训练 |

这就是一个完整的大模型训练流水线。在后续课程中，我们会逐一深入每个阶段。

### 9. 兼容主流框架

MiniMind 虽然代码简洁，但它兼容主流的模型推理和部署框架：

| 框架 | 用途 | 如何使用 |
|------|------|---------|
| **transformers** | 模型加载、推理 | 直接用 `AutoModel.from_pretrained()` |
| **vLLM** | 高性能推理服务 | 转换格式后加载 |
| **Ollama** | 本地模型运行 | 通过 GGUF 格式导入 |
| **OpenAI API** | API 服务 | 运行 `serve_openai_api.py` |

这意味着你可以把 MiniMind 当作学习项目理解完架构后，轻松迁移到生产级框架。

### 10. 面试中如何介绍 MiniMind

当面试官问"你有没有做过大模型相关的项目"时，你可以这样回答：

> "我深入学习了 MiniMind 这个开源项目——这是一个仅 64M 参数的大语言模型，采用和 LLaMA 相同的 Decoder-Only 架构（Pre-RMSNorm、SwiGLU、RoPE、GQA）。我不仅阅读了全部源码，还亲手完成了从预训练到 SFT 到 DPO 的完整训练流程。
>
> 选择 MiniMind 的原因是：它完全用 PyTorch 原生实现，没有依赖 HuggingFace Trainer 等高层封装，所以我能真正理解训练的每一步——数据加载、前向传播、损失计算、混合精度、梯度累积、分布式训练。
>
> 通过这个项目，我掌握了 [具体你学到的技术点]..."

**关键要点**：

1. 强调"深入理解"而非"简单使用"
2. 具体说出技术细节（GQA、RoPE、SwiGLU 等）
3. 说明为什么选这个项目（小而完整、纯原生、学习友好）
4. 结合你实际做的实验和踩过的坑

---

## 📂 对应 MiniMind 源码

| 概念 | 对应文件 | 关键代码 |
|------|---------|---------|
| 项目配置 | `model/model_minimind.py` | `class MiniMindConfig` — 所有超参数定义 |
| 模型架构总览 | `model/model_minimind.py` | `class MiniMindForCausalLM` — 完整模型入口 |
| 预训练脚本 | `trainer/train_pretrain.py` | 完整的预训练流程（数据加载、训练循环、保存） |
| SFT 微调 | `trainer/train_full_sft.py` | 有监督微调的训练流程 |
| 数据加载 | `dataset/lm_dataset.py` | `PretrainDataset`, `SFTDataset` 等数据集类 |
| 推理评估 | `eval_llm.py` | 加载模型 + 自回归生成 + 交互式对话 |
| Web 界面 | `scripts/web_demo.py` | Streamlit 对话界面 |
| API 服务 | `scripts/serve_openai_api.py` | OpenAI 兼容的 API 接口 |
| 模型转换 | `scripts/convert_model.py` | 将模型转换为 GGUF 等格式 |
| 依赖管理 | `requirements.txt` | 所有 Python 依赖包 |

---

## 🎤 面试考点

### Q1: 为什么选择 MiniMind 来学习大模型？它有什么优势？

**答**：选择 MiniMind 有三个核心原因：
1. **参数量小（64M）**：一张消费级 GPU 就能完成全部训练流程，适合个人学习和实验。对比 GPT-3 的 175B，MiniMind 只有其 1/2700，但结构完整不缺任何核心组件。
2. **纯 PyTorch 原生实现**：不依赖 HuggingFace Trainer、DeepSpeed 等高层框架，训练循环完全手写。阅读代码能真正理解训练过程中的每一步（混合精度、梯度累积、分布式训练等），而不是只会调参数配置。
3. **全链路覆盖**：从分词器训练到预训练、SFT、LoRA、DPO、PPO、知识蒸馏、Agent 训练——大模型训练的每个阶段都有对应代码。同时架构与 LLaMA 高度一致，学会了可以迁移到主流大模型。

### Q2: MiniMind 的模型参数量是怎么计算出来的？

**答**：以 minimind-3（hidden_size=768, num_hidden_layers=8, vocab_size=6400）为例：

1. **Embedding 层**：vocab_size × hidden_size = 6400 × 768 ≈ 5M（与 lm_head 共享，只算一次）
2. **每个 Transformer Block**：
   - Q 投影：768 × 768 ≈ 0.6M
   - K 投影：768 × 384（因为 GQA，kv_heads=4）≈ 0.3M
   - V 投影：768 × 384 ≈ 0.3M
   - O 投影：768 × 768 ≈ 0.6M
   - FFN（gate + up + down）：768 × 2048 × 3 ≈ 4.7M
   - RMSNorm：768 × 2 ≈ 0.002M
   - 小计：≈ 6.5M/层
3. **8 层 Block**：6.5M × 8 ≈ 52M
4. **其他**（最终 RMSNorm 等）：约 0.001M
5. **总计**：5M + 52M + ... ≈ **64M**

关键点：(1) GQA 让 K/V 投影的参数量减半；(2) 所有 Linear 层没有 bias，节省了参数；(3) lm_head 和 embedding 共享权重。

### Q3: MiniMind 和 LLaMA 在架构上有哪些相同点和不同点？

**答**：

**相同点**：
- 都是 Decoder-Only 架构
- 都使用 Pre-RMSNorm（先归一化再做计算）
- 都使用 SwiGLU 作为 FFN 的激活函数
- 都使用 RoPE（旋转位置编码）
- 都使用 GQA（分组查询注意力）
- 都使用 Weight Tying（Embedding 和 lm_head 共享权重）

**不同点**：
- 规模差异巨大：MiniMind 64M vs LLaMA-7B/13B/70B
- MiniMind 的 vocab_size 只有 6400（LLaMA 为 32000+）
- MiniMind 提供了 MoE 版本，LLaMA 2 没有（LLaMA 3 系列有 MoE 变体）
- MiniMind 训练代码完全自包含，不依赖分布式训练框架

### Q4: 大模型训练的完整流程是什么？MiniMind 覆盖了哪些？

**答**：大模型训练的完整流程：

1. **数据收集与清洗** → MiniMind 提供了预处理好的数据集
2. **分词器训练** → `train_tokenizer.py`（BPE 分词器）
3. **预训练（PT）** → `train_pretrain.py`（学习语言知识，Next Token Prediction）
4. **有监督微调（SFT）** → `train_full_sft.py` / `train_lora.py`（学习对话格式）
5. **对齐训练（RLHF/DPO）** → `train_dpo.py` / `train_ppo.py`（对齐人类偏好）
6. **评估与部署** → `eval_llm.py` / `scripts/` 下的各种脚本

MiniMind 覆盖了上述全部阶段，是目前少有的"全链路"开源教学项目。

### Q5: 如何快速估算一个模型需要多少 GPU 显存？

**答**：粗略估算公式（用于训练）：

- **模型参数**：每个 FP32 参数占 4 字节
- **梯度**：与参数等大，占 4 字节
- **优化器状态**：Adam/AdamW 为每个参数维护 2 个状态（一阶动量 + 二阶动量），各占 4 字节
- **激活值**：取决于 batch size 和序列长度

FP32 训练总显存 ≈ 参数量 × (4 + 4 + 8) = **参数量 × 16 字节**

MiniMind (64M)：64M × 16B ≈ 1GB（加上激活值等约 2-4GB，消费级 GPU 轻松搞定）
LLaMA-7B：7B × 16B ≈ 112GB（需要 2-4 张 A100 80GB）

使用混合精度（BF16/FP16）可以减半参数和梯度的显存，使用 LoRA 可以大幅减少需要更新的参数量。

---

## ✅ 自测题

### 题目 1（选择题）

MiniMind 的核心模型定义代码在哪个文件中？

A. `trainer/train_pretrain.py`  
B. `model/model_minimind.py`  
C. `eval_llm.py`  
D. `dataset/lm_dataset.py`  

<details>
<summary>查看答案</summary>

**B**。`model/model_minimind.py` 包含了所有模型结构定义：MiniMindConfig（配置类）、Attention、FeedForward、MiniMindBlock、MiniMindModel、MiniMindForCausalLM 等。

</details>

### 题目 2（选择题）

以下关于 MiniMind 的说法，**不正确**的是？

A. MiniMind 使用纯 PyTorch 原生实现，不依赖 HuggingFace Trainer  
B. MiniMind 的架构与 LLaMA 高度一致  
C. MiniMind 只支持预训练，不支持 SFT 和 RLHF  
D. MiniMind 支持 transformers、vLLM、Ollama 等主流框架  

<details>
<summary>查看答案</summary>

**C**。MiniMind 覆盖了完整的训练链路：预训练（train_pretrain.py）、SFT（train_full_sft.py / train_lora.py）、DPO（train_dpo.py）、PPO（train_ppo.py）、GRPO（train_grpo.py）、知识蒸馏（train_distillation.py）和 Agent 训练（train_agent.py）。

</details>

### 题目 3（简答题）

请列出 MiniMind minimind-3 模型的 5 个核心配置参数及其值，并解释每个参数的含义。

<details>
<summary>查看答案</summary>

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 768 | 隐藏层维度（d_model），决定了每个 token 的向量表示维度 |
| `num_hidden_layers` | 8 | Transformer Block 的层数，决定模型的深度 |
| `num_attention_heads` | 8 | Q 的注意力头数，决定从多少个角度计算注意力 |
| `num_key_value_heads` | 4 | K/V 的注意力头数（GQA），每 2 个 Q 头共享 1 组 KV |
| `vocab_size` | 6400 | 词表大小，决定模型能识别多少种不同的 token |

额外可提及：`intermediate_size`（约 2048，FFN 中间层维度）、`max_position_embeddings`（32768，最大序列长度）。

</details>

---

## 🔮 下一节预告

Phase 1 的基础知识到此结束！你已经了解了大语言模型的原理、Transformer 的架构、PyTorch 的基本用法，以及 MiniMind 项目的全貌。

在 Phase 2 中，我们将**深入源码**——从 MiniMind 的 Tokenizer 开始，逐个模块拆解模型代码，直到你能完全理解每一行代码的含义。

👉 **Phase 2 第一课：Tokenizer 分词器——如何把文字变成数字**
