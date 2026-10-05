# MyModel 设计书：从零训练小型文本模型

> 审查重写：2026-10-05。本文是实施方案，参数尚未经过本项目训练验证。
> 多模态扩展见 [DESIGN_MULTIMODAL.md](DESIGN_MULTIMODAL.md)。外部资料及本地参考版本见 §12。
> 数量约定：B = 10⁹，M = 10⁶；GB 为十进制，GiB 为二进制。所有训练 token 预算均用本项目最终 tokenizer 重新计数。

## 0. 可行性结论与范围

**在 8×A800 40GB 上，从零训练约 478M 的文本模型可行；原方案把研究配方、未经验证的效果承诺和工程需求混在了一起，不能直接按原预算执行。** 主要风险是训练时长、数据准备与恢复能力，以及小模型是否已经具备可供 RL 提升的初始能力。

项目定位保留为个人作品集：自己实现模型、数据管线、训练和推理，形成可以复现和解释的结果。质量目标是中英文基础对话、有限的数学与代码能力，不承诺接近充分训练的 MiniCPM5 或 Qwen3。

| 优先级 | 交付范围 | 启动条件 |
|---|---|---|
| 核心交付 | tokenizer → tiny 验证 → 478M、30B 预训练 → 短上下文 SFT → CLI / Web demo → 独立评测 | 算力、数据访问与恢复存储已落实 |
| 第一扩展 | 有效的可验证 RL；目标实现 JustRL II，先用 GRPO 建立对照 | SFT 在选定训练题上有非零且非饱和的正确率 |
| 按预算扩展 | 100B 预训练、8K / 16K / 32K 扩展、Agent SFT、LoRA、双专家 OPD | 前一级验收通过，成本已实测 |
| 多模态扩展 | 图像理解、语音输入、语音输出；图像生成单独立项 | 可用的文本 SFT checkpoint 已验收，无需等 OPD |
| 本轮不做 | MoE 主线、128K 质量承诺、全双工语音、多算法逐个展示 | 可另立研究分支 |

先用 0.1–1B tokens 验证 478M 的稳定性和吞吐，再承诺 30B。100B 是扩容目标，不是无条件的最低交付要求。未经验证的阈值和采样比例在本文均是**项目起始配置**，不称为 MiniCPM5 官方配方。

### 0.1 原设计需要纠正的关键点

| 原判断 | 审查结果与修订 |
|---|---|
| tiny 约 11.6M，而 embedding 就有 12.6M | 实际约 **15.93M**，见 §2 |
| μP 输出头置零，同时与 embedding 共享权重 | 会连输入 embedding 一起置零；原 μP 的 Adam 学习率缩放表也不正确。默认先用标准参数化，μP 单独验证 |
| GQA 8:1 质量无损、MoE 只需一个开关 | 都没有这种保证。GQA 需要评测；MoE 涉及容量、路由与权重迁移，不列为主线 |
| MiniCPM5 都用 JustRL II，q/k RMSNorm、FP8、μP 已全部对齐 | 1B 与 2B 的公开配方不同；2B 权重索引没有 q/k norm；发布 dtype 不能证明训练精度，MiniCPM4 的做法不能直接当作 MiniCPM5 的全部内部配置 |
| 400B SFT tokens ÷ 样本数 = 平均 4 万 tokens | 累计训练用量不是唯一语料体积，不能这样推导样本长度或采样率 |
| Mid-training 就是 decay；训完 30B 任意续训等价于一次训完 | 两者分别是训练阶段与 LR 调度；续训等价还依赖优化器、LR、数据顺序及 RNG 状态 |
| 32K 扩展只改 config；YaRN 自动得到可用 128K | 需要长样本训练、位置编码验证与多任务评测；128K 仅保留为实验 |
| RLVR 没有 reward hacking；OPD 没有能力干扰 | 验证器也可能被利用；蒸馏也可能退化，必须独立评测 |
| memmap 零内存、每个阶段只需 1% 算力 | page cache 和随机 I/O 有成本；RL 的 rollout、筛选与 critic 成本需单独测量 |

## 1. 硬件、时间和存储前提

| 环境 | 已知配置 | 用途 |
|---|---|---|
| 本地 | RTX 3060 6GB，16GB RAM | 短序列、micro-batch 1 的 tiny smoke test；CPU 数值检查 |
| 服务器 | 8×A800 40GB，1TB RAM，150GB SSD | 正式训练与 GPU 数据预处理 |

本地仍需要预算显存、内存和小数据集，不能因为只做验证就忽略资源限制。tiny 搜索的性能也不能从服务器峰值 FLOPs 直接推算成“分钟级”。

### 1.1 训练成本口径

设 `L` 为层数、`d_attn = n_heads × head_dim`，`P_mat` 为参与矩阵乘法的权重数。对于完整长度 `T` 的密集注意力，前向加反向可作如下粗估：

```text
F_train / token ≈ 6 × P_mat + 12 × L × T × d_attn
P_mat = Transformer 线性层参数 + lm_head 参数
```

共享 embedding 不会免除 lm_head 的计算。上式使用完整方阵的注意力估算；causal / 文档分块内核可能省去部分计算，而激活重计算、softmax、通信与数据等待会增加实际耗时。因此它是规划口径，不是精确的硬件执行量。

本配置 `P_mat = 478,150,656`，结果如下：

| 序列长度 | FLOPs / token，约 | 100B tokens FLOPs，约 | 注意力在该估算中的占比 |
|---|---:|---:|---:|
| 4096 | 4.480×10⁹ | 4.480×10²⁰ | 36% |
| 8192 | 6.090×10⁹ | 6.090×10²⁰ | 53% |
| 32768 | 1.575×10¹⁰ | 1.575×10²¹ | 82% |

约 7.3K tokens 时两项相当，但这不是 causal FlashAttention 或文档分块实现的固定交叉点。保留“短上下文完成大部分预训练”的决策。

**排期以实测端到端 tokens/s 为准。** 计入梯度累积、保存、评测与数据等待，排除启动编译后再测至少 100 个 optimizer steps。

| 8 卡实测总吞吐，示例情景 | 30B 纯训练时间 | 100B 纯训练时间 |
|---|---:|---:|
| 60,000 tokens/s | 5.79 天 | 19.29 天 |
| 120,000 tokens/s | 2.89 天 | 9.65 天 |
| 240,000 tokens/s | 1.45 天 | 4.82 天 |

以上不是预期性能；用真实测量替换，并另外预留数据准备、重跑和停机时间。MFU 必须说明显卡具体型号、非稀疏峰值、FLOPs 口径，不能混用不同硬件峰值。RL、SFT、多模态分别测量，不能沿用预训练吞吐。

### 1.2 显存预算与默认并行

默认采用 **fp32 参数及 AdamW 状态 + bf16 autocast**，避免 bf16 参数直接交给普通 AdamW 导致状态精度与预期不符。fp32 参数、梯度、两个 Adam 状态合计约 16 bytes/参数，即 **7.65GB / 卡**；激活、临时 bf16 权重、通信桶和 logits 另计。

| 项目 | 决策 |
|---|---|
| 并行 | DDP 首选；8 卡各保留完整模型与优化器，不除以 8 |
| 梯度累积 | 中间 micro-step 使用 `no_sync()`，optimizer step 才同步 |
| 内存不足 | 先缩小 micro-batch、开启激活重计算和分块 loss；再测试 ZeRO-1 / FSDP |
| 注意力 | PyTorch SDPA / FlashAttention；确认实际使用的 kernel |
| 精度 | A800 用 bf16，不要求 FP8 Tensor Core |

DDP 会用梯度桶与反向传播重叠通信，并非一定“最后只通信一次”。小模型不默认使用 ZeRO-3，但也不宣称其必然不可用。

**大词表 logits 是显存风险。** 单样本 `32768 × 49152` logits 就有约 3GiB bf16 / 6GiB fp32。长上下文 SFT 和 OPD 必须验证 fused 或支持重计算的 chunked linear-cross-entropy / KL；单纯先算完整 logits 再切块不会节省峰值显存。

### 1.3 150GB SSD 的执行条件

30B 和 100B 的 uint16 tokens 分别是约 60GB、200GB，尚未计索引、原文和 checkpoint。**100B 和多模态依赖远端存储或额外磁盘**。远端容量、可写权限、带宽与费用尚未在现有设计中落实，正式大规模训练前必须填实。

建议工作区上限：环境 30GB、数据缓存 30GB、恢复 checkpoint 与原子写临时文件 25GB、当前权重和日志 10GB、预处理暂存 15GB，保留至少 25GB 空间。以实际可用空间为准，不能只看 SSD 标称容量。

数据按不可变 shard 组织，预取下一批。至少保留多个领域的活跃 shard，按混合比例选择，再在 shard 内顺序读取并局部 shuffle；不要训完整个英文块、删掉，再训中文块。删除前核实 checksum、训练 cursor 和远端副本。多轮 SFT 也需要可重复读取的数据，不能“训过即删”后无法进入下一 epoch。

## 2. 文本主干与参数设置

### 2.1 默认结构

```yaml
model_type: mymodel                    # 自定义结构，不伪装成标准 Llama
hidden_size: 1024
num_hidden_layers: 32
num_attention_heads: 16
num_key_value_heads: 2
head_dim: 64
intermediate_size: 3584
vocab_size: 49152
hidden_act: silu
rms_norm_eps: 1.0e-6
qk_norm: true                         # 来自 minimind 风格，需做消融
rope_theta: 5000000                   # 候选基座，正式训练前固定
max_position_embeddings: 4096         # 首阶段经过训练的上下文
rope_scaling: null
tie_word_embeddings: true
attention_bias: false
mlp_bias: false
dropout: 0.0
ffn_type: dense
parametrization: standard
```

Block 使用 Pre-RMSNorm、RoPE、GQA、SwiGLU 和两条残差；最后加 RMSNorm。q/k RMSNorm 作用于每个 head 的 `head_dim`，权重在 heads 间共享，放在 RoPE 前。RMSNorm 的均方计算使用 fp32。

初始化起点：线性层与 embedding 的正态标准差 0.02，残差输出投影 `o_proj` / `down_proj` 可缩小到 `0.02 / sqrt(2L)`；norm 权重为 1。初始化是否改善稳定性在 pilot 中判断，选定后记录，不在长跑中随意切换。

GQA 16:2 减少 KV cache 与 K/V 参数，但不保证质量无损。保留 16:4 对照配置；这会增加约 8.39M 参数。q/k norm 与 RoPE 基座也都是结构选择，不是“照搬官方就必然正确”。为控制实验量，先搜 LR，再做少量结构对照，避免全组合搜索。

### 2.2 可复核的参数账

令 `d = hidden_size`、`a = n_heads × head_dim`、`b = n_kv_heads × head_dim`、`f = intermediate_size`：

```text
P_layer_linear = 2da + 2db + 3df
P_layer_norm   = 2d + 2head_dim         # 包含 q/k norm
P_total        = L(P_layer_linear + P_layer_norm) + d + Vd
```

| 模块 | 参数量 |
|---|---:|
| 32 层 attention 投影 | 75,497,472 |
| 32 层 FFN | 352,321,536 |
| 全部 RMSNorm，包括 q/k norm | 70,656 |
| 共享 embedding / lm_head | 50,331,648 |
| **总参数** | **478,221,312 ≈ 478.22M** |

| 配置 | hidden / layers / heads / kv / head_dim / FFN | 参数量，V=49152、tied、含 q/k norm |
|---|---|---:|
| tiny | 256 / 4 / 8 / 1 / 32 / 896 | **15,927,808** |
| small，可选 pilot | 512 / 12 / 8 / 2 / 64 / 1792 | **66,074,624** |
| base | 1024 / 32 / 16 / 2 / 64 / 3584 | **478,221,312** |

tiny embedding 占约 79%，适合验证实现，不是 base 的效果或超参代理。去掉 q/k norm、改变词表或 untie 都要重新计算，最终由独立参数枚举确认共享权重只统计一次。

### 2.3 μP 和 MoE 的边界

主线先用标准参数化。μP 保留为单独的、从初始化开始的实验，不能在已训练的标准 checkpoint 上切一个开关就当作 μP 模型。

若开展 μP：用官方 `mup` 包及 coordinate check 验证参数组、readout 缩放和初始化。官方 MuAdam 对双无限维隐藏矩阵的 LR 存在 `1 / width_mult` 缩放，不能写成“隐藏层 LR 都不变”。共享 embedding 需要专门的共享 readout 方案；为简化首个实验，可用 untied 输出头并单独预算参数。不得把 tied 输出头 zero-init。

宽度实验固定深度、head_dim 和结构关系，记录 base shapes。验证激活与更新坐标量级随宽度稳定，再验证有限宽度的 LR 排序；不同宽度 loss 曲线无需重合。μP 不保证任意深度、batch、GQA / norm 改动都可直接迁移。ScalingBench 是可选研究，需自行证明条件 loss 与本项目下游分数有关，不能拿已有论文的 sigmoid 曲线作为本项目验收事实。

MoE 暂缓。将来需要独立解决专家初始化、总参数与激活参数、路由容量、aux loss、分布式通信和 dense checkpoint 转换，不把它描述为低成本 FFN 开关。

## 3. Tokenizer 与协议

**一次性训练中英混合 byte-level BPE，总词表 49152。** 特殊 token 先进入训练器，例如占前 128 个 ID；其余为统一学习的 BPE，不划出未经实现验证的“32K 通用 + 17K 中文补充”区间。

| 项目 | 起始方案 |
|---|---|
| 训练语料 | 从 train split 抽取约 2–5GB 清洗文本，覆盖中英、代码、数学；中文适度提高抽样权重 |
| 词表候选 | 32K 与 49152 在小语料上比较；正式预训练前选定，默认 49152 |
| 必须覆盖 | byte alphabet，中文标点、代码缩进、数字与数学表达式 |
| 协议 | 独立 PAD、文档 EOS、消息开始 / 结束；think / tool / image / audio 控制符及预留符 |
| 模板 | versioned chat template；明确 thinking 模式、工具 JSON schema 和 assistant span |

比较中英 chars/token、代码 bytes/token、往返解码、词表利用率与短任务性能。49K 是否划算由测量决定，不使用未经测量的“中文平均 1.5 或 2.5 字/token”作为固定事实。

输出 `tokenizer.json`、merges、special token 映射、chat template 和 manifest 哈希。配置从该 manifest 读取 ID；训练开始后冻结 BPE merges 与既有 ID。预留符进入 vocab 也必须被正确声明为单 token，而不只是一个 Python 常量。

纯文本 `vocab_size=49152` 可用 uint16 存储。多模态如追加 16384 个 image codes，总行数 65536，最大 ID 65535，仍可用 uint16；超过该值则需要 uint32。label 的 `-100` 不存进 uint16，单独存 mask，训练时生成 int64 labels。

## 4. 数据来源、计数与质量

### 4.1 已核实的候选来源

下表的外部 token 数采用各来源自身 tokenizer / 版本口径，只表示候选池大小，不等于本项目可直接使用的 token 数。

| 数据 | 已核实信息 | 本项目用途 |
|---|---|---|
| [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | 同一仓库含 `en` / `zh` splits；当前卡片约 1T 英文、120B 中文；论文 Table 1 英文写 1800B，两者口径不同 | 主体网页预训练 |
| [Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3) | 当前卡片英文 400B+、中文 200B+；不能拿早期论文英文 200B 当作当前唯一版本 | 精炼数据 pilot / decay |
| [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | L2 约 400B、L3 约 150B，11 种语言 | 代码预训练与能力强化 |
| [UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) 及其 L2 / L3 仓库 | 卡片分别约 170.5B、33.7B、88B；L2 / L3 是独立 repo，不能默认都在 L1 repo 中 | 数学预训练 |
| [UltraX-Preview](https://huggingface.co/datasets/openbmb/UltraX-Preview) | 公布的预训练候选，需探查内容与可得子集 | 可选替换网页源 |
| [UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) | **15,036,178 条**，有 `think` / `no_think`；访问文件需登录接受条件，本次只读取公开卡片和元数据 | 文本 SFT 首选 |
| [UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609) | **483,661 条轨迹**，不是独立任务数 | 可选 Agent SFT |
| [UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) | **85,995 条**，数学 / 代码 / 长上下文 / 知识 | RL 中可验证且难度适配的子集 |

UltraData L0–L4 是数据管理层级，不等于所有 L3 样本都更适合当前模型，也不要求机械地按 L1→L2→L3 跑。论文中分层训练的 1.49 个百分点增益属于论文的受控实验，不能承诺本项目得到相同增益。

### 4.2 数据准备的必需产物

正式下载前，对每个源探查小批样本，确认 schema、语言、质量字段、文本 / 测试用例路径、授权状态与实际可用版本。数据 manifest 至少记录：

```text
repo_id、revision、config、split、文件 checksum、license / 上游约束
清洗版本、采样种子、tokenizer hash、唯一文档 / token 数
实际训练曝光 token 数、领域与语言分布、长度分位数、过滤原因
```

建立源级 / 文档级 train-dev-test 划分，**再训练 tokenizer 和构造训练缓存**。同一原文及其改写、同一题目变体、同一代码仓库不能跨 split 泄漏；做精确去重及适量近重复去重，检查评测题污染。公开 SFT 和 RL 数据的 benchmark 去污染声明不能替代本项目自己的检查。

数据卡的许可证不能自动覆盖全部上游材料。记录来源、使用与发布条件。SFT 文件访问若暂不可用，可用 minimind 文档指向的开放小数据或其他已核实的开放对话子集跑流程；报告中标注替换来源与规模，不宣称仍与 MiniCPM5 同数据。本次没有接受 gated 条件或下载完整数据集。

### 4.3 预训练混合

按**本项目 tokens**控制混合，起点如下，作为实验超参：

| 领域 | stable 阶段 | 30B 路线对应量 |
|---|---:|---:|
| 英文网页 | 45% | 13.5B |
| 中文网页 | 25% | 7.5B |
| 代码 | 20% | 6.0B |
| 数学 | 10% | 3.0B |

这里的中文网页占比不等于全语料中文占比；代码注释与数学文本还包含中英文。比例通过领域内清洗、抽样、tokenize 后的真实计数控制，限制少量高分样本重复曝光。

decay 可提高 L3 份额，但先保持领域比例做一次同预算对照，区分“质量层级变化”和“领域调权”两种效果。若同时改比例，必须单列实验。30B 不要求完全不重复；记录唯一 tokens、训练曝光 tokens 和重复倍数，以去重后的验证结果判断收益。

### 4.4 预 tokenize、packing 和 attention

预训练离线编码为 uint16 shards，配文档 offsets、长度、来源和质量 metadata。`np.memmap` 不一次性加载整库，但 OS page cache 会占内存，随机窗口也会产生 I/O。

默认采用 **EOS 分隔 + document-aware packing**：

- 文档 / 长文切片内部用 causal attention，不同文档不互相注意；position IDs 按片段定义并记录。
- 优先使用 varlen FlashAttention / 支持分块的 kernel，不构造长序列的完整 `T×T` 布尔 mask。
- 预测 EOS 的位置参与训练；EOS 之后预测下一篇文档首 token 的跨文档目标设为 ignore。
- 长文切片的重叠 / 起点和不完整窗口策略固定，避免随机采样悄悄重复大量数据。
- 若本地仅实现普通 EOS 拼接作为基线，应明确它允许跨文档 attention，不能称为文档隔离，也不能把它一概称为错误实现。

数据加载器做确定性的 rank 分片、局部 shuffle 与领域采样，保存真实的 shard / offset / sampler 状态。不能一边随机有放回抽窗口，一边声称已经严格保证小于一 epoch。

### 4.5 SFT：采用适配小模型的分布

官方公开数据含 Math、Code、Knowledge、Chinese-general、IF、Multi-lang-Math、Multi-lang-Knowledge 七个 config。其中 Math / Code 各约 550 万 / 579 万条，天然分布偏重推理。**公开池比例不等于 MiniCPM5 内部训练曝光比例。** deep-thinking 与 hybrid-thinking 的训练用量也不等价于 `think` / `no_think` 文件的 1:1 抽样。

本项目先训练短答案和基础指令，再引入完整、较短的推理。首轮目标 **0.2–0.5B 输入 tokens**，约 5–20 万条候选，按实测长度确定条数；上限可增至 1B，不默认全程 32K 和多 epoch。

| 领域 | assistant 有效监督 tokens 的起始比例 |
|---|---:|
| Chinese-general | 35% |
| IF | 20% |
| Knowledge | 15% |
| Code | 15% |
| Math | 15% |

以中英为目标，筛去暂不支持的其他语言子集。第一轮有效回复监督以 no_think 约 80%、短 think 约 20% 为起点，根据准确率和长度调整；这是自己的行为目标，**不是官方等比例复刻**。同时报告样本、输入 tokens、有效回复 tokens 三种分布。

正确顺序：探查并保存原始长度 / 领域统计 → 质量与长度筛选 → 统计可用池 → 目标分层抽样 → 复核最终曝光分布。默认窗口 4K；超长样本可整条排除，或按完整轮次 / 独立任务拆分。不得截断 assistant 推理和答案；若某领域被大量排除，记录排除比例并补入短样本，不能声称过滤后分布天然保持不变。

## 5. 训练路线和优化设置

### 5.1 分阶段路线

```text
代码 / 数据验证
  → 478M pilot（0.1–1B，4K）
  → 30B Base（4K，WSD）
  → 短上下文 SFT → demo / 固定评测
  → 可验证 RL：GRPO 对照 → JustRL II

独立分支：stable checkpoint → 扩至 100B → 重做对应 SFT / 后训练
按需分支：8K → 16K / 32K；Agent SFT；LoRA；双专家 OPD；多模态
```

文本 SFT 跑通后就能开始多模态。LoRA 是任务适配分支，**不要求串在通用 SFT 与 RL 之间**。Agent 数据只在需要工具能力时加入，数学 / 代码 RL 不以 Agent SFT 为前提。

### 5.2 标准参数化的预训练起点

| 项 | 起始值与约束 |
|---|---|
| 优化器 | AdamW，β=(0.9, 0.95)，eps=1e-8 |
| 峰值 LR | pilot 比较 3e-4、6e-4、1e-3；默认候选 6e-4，不套用 μP 的 7e-3 |
| weight decay | 0.1；norm / bias / 共享 embedding 组不衰减，记录分组并对共享参数去重 |
| 梯度裁剪 | global norm 1.0，所有累积及同步完成后裁剪 |
| 全局 batch | 起点 524,288 输入 tokens / optimizer step |
| 示例 batch | 8 卡 × micro-batch 2 × 累积 8 × 4096 = 524,288，需显存实测 |
| warmup | 起点 0.3B tokens，按已消费 tokens 调度，不按 micro-step |
| 精度 | fp32 optimizer / 参数状态，bf16 autocast；bf16 不默认启用 GradScaler |
| loss | 对实际有效 tokens 归一化；padding / ignore 不进入分母 |

DDP、变长 SFT 和梯度累积必须按整个 optimizer step 的**全局有效 token 数**归一化，避免各卡独立 mean 后使短样本权重变大。若 DDP 对梯度取 rank 平均，每个 micro-batch 使用 `local_loss_sum × world_size / global_valid_tokens_in_optimizer_step`，共享该步分母，不再额外除以累积步数；等价的更新后缩放方案也需验证。loss shift 只做一次：位置 `t` 的 logits 预测位置 `t+1` 的 label。

LR pilot 采用独立短预算，warmup 取该预算的 1–3%，保证真正比较到峰值 LR。选定配置后正式长跑重新初始化并使用表中的 0.3B warmup；pilot 算力另外计入总预算，不把尚未达到峰值 LR 的短跑当作稳定性结论。

### 5.3 WSD 与可扩容 checkpoint

定义具体调度函数，而不是只写 WSD 名称：线性 warmup → 常数 stable → 最后 10% 预算的线性 decay 至峰值的 10%。其他 decay 曲线作为消融，不与默认混写。

```text
30B 路线：warmup 0.3B + stable 26.7B + decay 3B
100B 路线：warmup 0.3B + stable 89.7B + decay 10B
```

在累计 **27B、尚未 decay** 时保存可恢复的 stable checkpoint。先从它分支做 3B decay 得到 30B Base，再测试下游；以后扩到 100B，从 27B stable 的完整状态继续。这样最终主分支累计 100B，但探索分支还额外消费了 3B，GPU 总预算必须加上这部分及重复下游训练。

从 decay 完成的 30B checkpoint 重新升 LR 也是续训实验，但不等价于原 stable 延长。SFT / RL 权重不能当作原预训练分支直接接回去。Mid-training 指目标能力 / 数据分布的继续训练，可能搭配 decay 或重新 warmup，不与调度名称绑定。

### 5.4 上下文扩展

先完成 4K 交付。若需要长文能力，做 8K pilot，再选择 16K / 32K；每一级重新测 micro-batch、attention kernel、重计算和短任务回归。

| 扩展 | 初始输入 token 预算 | 是否直接承诺 |
|---|---:|---|
| 4K → 8K | 0.1–0.3B | 通过评测再扩大 |
| 8K → 16K / 32K | 0.5–1B pilot，收益明确时扩至 3–5B | 数据、收益与时间允许才执行 |
| 32K → 128K | YaRN / 位置缩放研究 | 不作为必达能力 |

使用真实长文、仓库级代码、合法可用的书籍与长文问答，长度分桶，穿插短文本。位置编码方案比较原 RoPE 继续训练与有文献依据的缩放；不能仅由 `theta=5e6` 推导“必然无需缩放”。扩展后冻结方案，模型 config 区分 `trained_context_length` 与 `serving_max_length`。

MiniCPM4 报告 §5.1 的 4K→32K 阶段用了 20B tokens 和 **LongRoPE**，推理再用 YaRN；它同样是 8 倍扩展。原文将其写成 4 倍，并省略 LongRoPE，是错误的。

验收至少含多深度、多长度 NIAH、RULER 多任务、长文问答，以及短上下文回归。NIAH 只证明特定检索能力，不单独证明长文推理。预训练 Base 不善于遵守问答指令时，在相同 SFT 适配条件下比较扩展前后结果。

### 5.5 SFT、工具与 LoRA

| 阶段 | 起点 |
|---|---|
| 文本 SFT | LR 1e-5–5e-5，warmup 1–3%，cosine，先 1 epoch；按 dev 结果决定是否重复 |
| Agent SFT | 选 1–5 万条可复现的 Tool-Use 轨迹，混入普通 SFT 回放；不直接吞下全部 Office / Search 环境 |
| LoRA | rank 8 / 16，alpha 16 / 32，LR 1e-4–3e-4，明确注入模块及 merge 验证 |

SFT label mask 优先由模板的 assistant spans 构建，并核对 role / span 边界；不要只扫描文本字符串，以免用户内容中出现标记时误计 loss。assistant 内容、工具调用和消息结束符参与监督；user、system、tool 返回、padding 默认 ignore。工具返回虽不算 loss，仍作为上下文输入。

预处理完成后缓存 tokenized SFT，可保留原始 JSONL 以复查。增强只使用受控、记录种子的模板变化；不能随机删 think 标签同时保留不匹配的内容。推理时的 think 开关、工具 schema 必须与训练协议一致。

Agent 只支持自建的少量确定性工具，例如计算器、受限查询、受限代码测试。验收既看调用格式，也看参数正确率、读工具返回及最终任务成功率。合法 JSON 不代表会用工具。

## 6. 可验证 RL：先验证信号，再复现 JustRL II

### 6.1 数据和预算门槛

从 UltraData-RL-2609 的短数学 / 代码子集起步，探查 100–500 道 dev 题，每题采样 4–8 次，估计当前 SFT 的 pass rate。完整池中数学 32,412、代码 23,665、长文 18,046、知识 11,872 条；其难度为更强 checkpoint 校准，不能假定适合本项目。

主训练先选 1–5 千道有可靠 verifier、组内出现对错差异的题。若大多数全错，先补基础题或短推理 SFT；延长到 8K 不会自动产生有效 RL 信号。

| 超参 | 本项目起点 |
|---|---|
| 每题 rollout 数 | 4，目标对照再扩到 8；不默认 16 |
| 回复上限 | 1024–2048 tokens，实测有收益再扩到 4096 / 8192 |
| 温度 / top-p | 0.8–1.0 / 1.0，评测固定另一个配置 |
| 每次更新题目数 | 起点 16–32 个保留组；按显存拆 micro-batch |
| actor LR | 1e-6 起，少量比较至 5e-6 |
| critic LR | 5e-6–1e-5 起，单独配置 |
| 初轮规模 | 100–300 optimizer steps，按端到端时间与 dev 改善决定是否继续 |

每条轨迹的 `prompt_tokens + max_new_tokens` 不得超过已验收窗口。4K 模型按实际 prompt 长度分配剩余回复预算；提示过长时完整排除或重构任务，不截掉题目条件。扩到 4096 / 8192 的回复上限前，先验收能容纳 prompt 和回复的更长上下文，并重新统计被排除的任务分布。

记录 sampled / retained prompts、生成 token 数、全对 / 全错组比例、rollout 与训练时间。动态采样设置候选组上限，例如目标组数的 4 倍，超过上限报告信号不足，不无限生成。RL GPU 预算单列，先预留可用 GPU 小时的 10–20% 做试验，不能把“预训练 FLOPs 的 1%”当成足够的承诺。

### 6.2 GRPO 对照

使用同一 SFT 初始化、prompt 池、verifier 和生成预算，建立无 critic 对照。起始采用组内 reward 中心化：`A_i = R_i - mean(R_group)`；是否除以组内标准差是独立配置，默认不除，与本次查阅的 JustRL II baseline 描述保持一致。

response mask 只覆盖真实生成 tokens / EOS，不包括 prompt 和 PAD。old log-prob、advantage detach；记录 clipping ratio、entropy、KL、长度、截断率与独立 dev pass@1。训练 reward 上升不构成验收。

### 6.3 JustRL II 的必要实现约定

参考公开方法与代码 [S6]；以下是缩短上下文后的项目复现，不冒充官方 128K 完整规模。

1. **独立 critic**：复制 policy 的 backbone 和输入 embedding，去掉 LM 输出头，接标量 value head；不与 actor 共用可训练 backbone。value head 权重初始为零，bias 为本项目测得的 raw reward 均值，不能硬套官方 0.5 / 0.52。加载 actor checkpoint 后重新校验该头；恢复 critic 自己的 checkpoint 时保留已训练值。
2. **critic 预热**：冻结 actor，只用其 rollout 训练 critic。30 步是参考代码起点；同时检查 held-out within-group AUC 高于 0.55、explained variance 为正及校准误差，未达标不启动 actor 更新。
3. **长度自适应 GAE**：`gamma=1`，`lambda_i = k ** (1 / response_length_i)`，起点 `k=0.5`。lambda、value、advantage 和 returns 使用 fp32，不能让 bf16 把接近 1 的 lambda 舍入为 1。
4. **动态采样和组结构**：选择有对错差异的组，保留同 prompt 多条轨迹用于 critic 诊断。其收益和丢弃 rollout 的成本都要报告。
5. **两种 reward 流**：critic 回归未经长度惩罚的正确性 reward；actor 的优势使用加长度惩罚的 reward，value 回归目标不使用该惩罚。

```text
r_raw,t = 0（中间步），最终步 = verifier 的正确性 reward
r_actor,t = 0（中间步），最终步 = R_raw + R_length
δ_t = r_actor,t + V(s_{t+1}) - V(s_t)
A_t = δ_t + lambda_i × A_{t+1}
critic_target_t = gamma=1、lambda=1 的 raw reward suffix sum
L_actor = - mean_masked(min(ratio × A, clip(ratio, 1-ε, 1+ε) × A))
ratio = exp(logp_current(action) - logp_old(action))
L_value = mean_masked((V - critic_target)²)
```

首版将回复触顶作为有限生成预算的 episode 结束，记录 `truncated=true`，其边界 `V(s_next)=0`；只有在已生成内容中提取到完整、可验证的答案才给 correctness reward，并另加尾部长度惩罚。正常 EOS 同样以 terminal 处理，服务故障则丢弃并重试；代码执行超时按 verifier 的失败规则记录。若后续采用可续接的部分 rollout，必须改为非 terminal 并正确 bootstrap，不能沿用上述边界。clip 起点 ε=0.2，不照搬长序列工程优化而不解释。

`lambda^L=k` 是设计记号；按上述 last-token 终奖索引，首 token 实际权重是 `lambda^(L-1)=k^(1-1/L)`。用短序列手算和参考实现确认索引，不用文档近似替代单元验证。

长度惩罚在接近上限的尾部缓冲区从 0 线性降到 -1。根据本项目 baseline 长度分布选择缓冲区，而非直接复制官方 128K 的 token 数。“先压长度再增准确率”是 MiniCPM5-1B 的公开安排，**不与 2B 的 JustRL II 混成同一个必需两阶段配方**。

参考博客 actor / critic LR 为 1e-6 / 1e-5，当前公开 release env 的 critic LR 为 5e-6。复现时锁定代码版本并说明采用哪组，不能把 MiniCPM4 的 1e-5、KL=0.001 当成 MiniCPM5-2B 的统一设定。参考 KL 惩罚若加入，应作为明确的稳定性扩展及消融。

### 6.4 验证器与计算成本

数学使用可靠答案抽取与数值 / 符号等价校验；代码在隔离环境中执行测试，限制时间、内存和输出，使用未暴露给模型的测试。为 verifier 构造正常、错误、歧义、超时、投机答案样本，分别统计。RLVR 仍可能有错误标注和 reward hacking。

actor、独立 critic、可选 reference、rollout engine 和 KV cache 分别预算显存。不能仅因参数不到 1B 就断言长序列多实例一定放得下。先使用同步 rollout / update，后续如果异步或复用旧轨迹，必须处理 policy version、old log-probs、重要性修正与同步时机。

成功标准：独立 dev 的准确率有稳定改善，长度 / 退化在约束内，能展示 critic 的校准及组内排序收益。比较等生成 token 或等 GPU 时间的 GRPO 与 JustRL II，不能把后者多消耗的 rollout 算力忽略。

## 7. OPD：条件满足后的双专家实验

先证明两个 RL 专家各自比共享 SFT 初始化有效，例如数学和代码，再做合并。首轮只做 **2 个专家**；没有有效专家时不为走流程而做 OPD。教师与学生不仅 vocab 大小相同，还必须 tokenizer、ID 映射、模板语义和参数化兼容。

学生在线生成回答，教师在相同学生前缀上提供分布。第一版采用明确可检查的固定前缀全词表 RKL 目标：

初轮 LR 1e-6–5e-6，回复最多 1024 tokens，prompt 加回复不超已验收窗口，先跑 100–300 步。rollout 使用 temperature=1、top-p=1，logits 比较使用同一温度；不复用已过多次更新的旧回答冒充当前 on-policy 样本。

```text
KL_t = Σ_v p_student(v | student_prefix_t)
             × (log p_student(v | student_prefix_t) - log p_teacher(v | student_prefix_t))
L = response 位置 KL_t 的有效 token 均值
```

teacher eval + no_grad，student 分布参与梯度；log-softmax / KL 累积使用 fp32。只在回复位置计算，按 prompt 领域选择教师，batch 内保持领域覆盖。按 token chunk 计算并验证峰值内存，避免保存两份全长全词表 logits。

这属于本项目的全词表 RKL 基线。若按 MiniCPM5 描述移植 RL advantage 形式，要使用逐候选 token 的 log-prob 差与配套策略更新，固定参考 policy，并验证梯度方向；**不能直接把一个与 action 无关的负 KL 标量广播给整条回答，认为已经传递了教师分布**。

top-k 并集版是后续近似：定义是否重新归一化、遗漏质量如何处理，并与小词表 full-KL 对照，不能称为精确全词表 KL。MiniCPM5-2B 的 16 专家 full-vocab 与 1B 的 top-k 并集是两种公开实现描述。

验收对照至少包含 SFT、两个专家、简单顺序微调、OPD；报告每领域 dev 分数、专家差距及通用能力回归。“接近专家”不是结构保证，结果不好也应解释并保留记录。

## 8. 工程组织、恢复和推理

### 8.1 最小代码组织

```text
MyModel/
  configs/                model/{tiny,small,base}，各阶段训练配置
  mymodel/model/          config、norm、rope、attention、block、lm
  mymodel/data/           tokenizer / protocol、manifest、packing、sampler
  mymodel/train/          supervised loop、schedule、checkpoint
  mymodel/rl/             rollout、verifier、GRPO、critic / GAE、OPD
  mymodel/infer/          generate、CLI、Web
  tools/                  数据探查、抽样、tokenizer、tokenize、评测
  tests/                  数值一致性和关键训练契约
  experiments/            固定配置、数据版本、指标与资源实测
```

这是计划结构，当前目录只有设计文档，不把上述脚本写成已交付。先实现核心交付，再新增研究模块。监督训练循环可以共享；RL 独立管理采样、旧策略、critic 和更新时序，不承诺所有算法只是一个 loss 插件。

### 8.2 checkpoint

区分 `init_from`（继承权重，开始新阶段）和 `resume_from`（完整恢复同一训练）。完整状态至少有模型、优化器、scheduler、global step、累计输入 / 有效 tokens、全部 rank RNG、sampler / shard cursor、配置和数据 manifest 哈希、代码版本。

本配置 fp32 参数 + Adam m/v、无梯度的恢复文件约 5.74GB；其他 master-weight 实现会更大，实际测量。保留 last / prev、少量可扩容 stable 里程碑和当前 bf16 export。原子写需要额外临时空间；远端备份**必须包含关键 stable checkpoint 的优化器状态**，不能仅发布 bf16 权重后删掉唯一续训状态。

保存先获得一致的 CPU snapshot，再异步写入 tmp、校验并原子 rename。限制同时在写的 snapshot 数；1TB RAM 不使拷贝或异步状态一致性变得免费。远端上传确认成功前不删除本地副本。导出 manifest 校验 tied storage、config 与 tokenizer。

### 8.3 推理与部署

实现 KV cache、causal mask、padding / position IDs、temperature / top-k / top-p 和 EOS。cache 追加时支持 `q_len != kv_len`，不依赖一个可能错位的 `is_causal=True` 假设。先测试未量化，再考虑低比特导出。

单序列 bf16 KV cache 公式：`2 × L × tokens × n_kv_heads × head_dim × 2 bytes`；本模型 4K 约 64MiB，32K 约 512MiB，128K 约 2GiB，另加权重和临时激活。GQA 实现不能在持久 cache 中把 KV 重复存成 16 heads。

CLI 和 Web 支持训练模板与能力说明。分别报告 prompt 长度、生成长度、batch、硬件、dtype 下的 TTFT、decode tokens/s、峰值显存，并与同档开源模型作外部参考；由于训练数据和 tokens 差距，不能把差距都归因于模型代码。

## 9. 验收与必要验证

| 阶段 | 必须留下的证据 | 未通过时的动作 |
|---|---|---|
| tiny | 16–32 条短样本可明显记忆；cache / mask / shift 数值一致 | 缩小任务、检查实现，不用“500 条 200 步不过必有 bug”诊断 |
| pilot | 无 NaN，loss / 梯度稳定，100 步吞吐与显存，恢复连续性 | 调整 LR / batch / 初始化后重跑 |
| 30B Base | train / dev loss、唯一与曝光 tokens、领域统计，中英续写 | 看去重、采样和语言覆盖，评估是否扩大预算 |
| SFT | 固定指令 / 多轮集、结构化约束、数学 / 代码短任务与长度分布 | 补短样本、修 mask / 模板，暂缓 RL |
| GRPO / JustRL II | dev pass@1、rollout 成本、截断 / entropy、critic AUC / EV | 信号不足先改难度；critic 不合格停止 actor 更新 |
| 长上下文 | 多长度 / 多深度检索、长文任务、短任务回归、TTFT | 留在已验证的上下文，不强开 128K |
| Agent / LoRA / OPD | 各自 baseline、独立任务分数、通用能力回归 | 不并入发布主权重，保留实验报告 |

核心评测包含语言 / 领域 dev NLL、至少 200 条独立人工核查的中英指令集、适合当前能力的数学与代码任务；可另跑 GSM8K、MATH500、MBPP / HumanEval、C-Eval / CMMLU 小子集。固定版本、prompt、采样和答案提取方法，报告样本数与不确定性。不同 tokenizer 模型的 raw loss 不直接横比；跨模型用下游准确率或 bits/byte 等可比口径。

必要的实现验证：

- RoPE、RMSNorm、GQA 与短序列 fp32 参考实现一致；bf16 使用合理容差。
- full forward 与 cache decode 一致，包含左 padding、多 token 追加及长 prefix。
- 文档 attention 隔离、boundary label mask、SFT role / EOS / tool mask 和 shift 正确。
- 梯度累积与全局有效 token 加权和等价大 batch 的梯度一致。
- resume 后下一批数据、LR 和短跑更新与连续训练相符；跨阶段加载没有无说明的 missing keys。
- GAE、EOS / 截断、长度惩罚与 critic raw targets 用手算短序列验证；RKL 符号、mask 与 teacher 无梯度正确。

这些是模型训练契约的必要验证，不要求为文档或每个简单开关写镜像测试。真实训练效果单独记录，不能由 tiny 过拟合或单条对话截图代替。

## 10. 执行顺序与停机规则

1. 固定代码与数据版本，核实 gated 访问、远端容量和可用 GPU 时段；探查源数据及 tokenizer。
2. 实现 tiny 模型、协议、监督训练、恢复、推理和独立评测；短样本验证通过。
3. 478M pilot，实测 LR 稳定性、吞吐、显存与 shard 流水，填写 30B 完整预算。
4. 运行 27B stable 分支与 3B decay，保留完整 stable 状态；完成短 SFT、demo 和评测。
5. 选可学任务做 GRPO / JustRL II 对照；依结果开展双专家 OPD，或从文本 SFT 开始多模态。
6. 有预算且存在清晰收益时扩到 100B / 长上下文，并重做所需下游阶段。

出现重复 NaN、恢复无法复现、数据污染或 verifier 失真时停止长跑并修复；dev 指标持续无收益时先诊断，不用增加 token 数掩盖问题。发布门槛与研究结果分开：可信的失败实验也能成为作品，但不能标为已获得的能力。

## 11. 交付物与尚待落实事项

交付代码、固定配置 / 数据 manifest、可恢复状态及阶段权重、资源实测、独立评测、demo 与实验解释。Base / SFT / RL 命名描述实际阶段；只有实际进行 mid-training 才发布相应档位。

正式大跑前需要落实：GPU 可连续使用时长与费用；远端存储与带宽；SFT gated 文件访问；源数据字段及短样本可用量；tokenizer 压缩率；base pilot 的 LR / batch / 吞吐；RL 子集初始 pass rate；评测回归容忍度。文档给出了默认路线，但没有把这些未知条件写成已成立的事实。

## 12. 核验依据与版本边界

以下链接于 2026-10-03 核验。官方事实、论文结果与本项目假设在正文分别表述；本次没有运行 GPU 训练，也没有下载完整训练数据。

| 编号 | 来源 | 本次使用的证据 |
|---|---|---|
| S1 | [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)、[config](https://huggingface.co/openbmb/MiniCPM5-1B/blob/87179e5c1f455ef22e6223592d2d61351b525bfc/config.json) | hidden 1536、24 层、heads 16 / KV 2、head_dim 128、vocab 130560、untied、theta 5e6；200B deep-thinking + 200B hybrid-thinking；1B RL 与 OPD 描述 |
| S2 | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)、[config](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/config.json)、[权重索引](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/model.safetensors.index.json) | hidden 2048、42 层、16 / 2、head_dim 128；索引没有 q/k norm / MTP 参数；400B deep-thinking，JustRL II 与 16 专家 full-vocab OPD |
| S3 | [MiniCPM4 报告](https://arxiv.org/abs/2506.07900) | μP / ModelTunnel / WSD；§5.1 LongRoPE 与 20B 长上下文阶段；Table 8 的 0.5B 训练约 1T tokens。报告内部不同章节的 stable / decay 总量表述有差异，不据此声称精确复制 MiniCPM5 |
| S4 | [UltraData 论文](https://arxiv.org/abs/2602.09003)及 §4 各数据卡 | 分级体系、表格规模及其版本差异；具体字段与可用量仍要用锁定版本的样本核验 |
| S5 | [UltraData-SFT-2605 卡片](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) | 15,036,178 条、七领域、think / no_think；revision `affda6aca75e7cff78e73f93ad08d4c3b01f097c`，gated=auto |
| S6 | [JustRL II 博客](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic)、[公开代码](https://github.com/merak0514/JustRL-II)、[方法说明](https://github.com/merak0514/JustRL-II/blob/main/docs/method.md)、[release env](https://github.com/merak0514/JustRL-II/blob/main/justrl2/configs/minicpm5-2b-math-128k.env) | 独立无 LM head critic、均值 bias 初始化、预热、fp32 长度自适应 GAE、raw / penalized reward 分离；参考完整训练为 16 节点 × 8 GPU，不照搬其算力规模 |
| S7 | [mup 官方仓库](https://github.com/microsoft/mup) | Adam 参数组缩放、base shapes、coordinate check 和 resume 注意事项 |
| S8 | 本地 [minimind 模型](../minimind/model/model_minimind.py)、[数据](../minimind/dataset/lm_dataset.py)、[训练](../minimind/trainer/train_pretrain.py) | commit `f659b55761b754d306bd140573493a6543cafd7f`；GQA / qk norm / SwiGLU / tied、assistant mask、DDP。其短序列教学配方不直接证明 478M 长跑可行 |

已查阅的 MiniCPM5 模型卡和配置没有给出可直接复用的完整预训练 token 数、精确混合权重和全部内部超参。本文据此采用可核实的结构与方法，明确本项目自己的预算与实现选择，不再用“配方全对齐”概括尚未证实的内容。
