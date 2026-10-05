# MyModel 设计书：在文本模型上扩展多模态

> 审查重写：2026-10-05。前置设计见 [DESIGN.md](DESIGN.md)。
> 本文定义图像理解、语音输入、语音输出及可选图像生成；各阶段均需真实训练和独立评测，当前不是已实现的能力。
> MiniCPM-o、SigLIP2、Mimi 资料于 2026-10-03 核验；数据候选需在实际使用前探查。本地参考版本和来源见 §11。

## 0. 可行性结论与修订边界

**冻结编码器 + projector 接入 478M 文本主干是可行的。独立 Talker 也可作为研究项目，但需要可靠的语音 codec、文本条件与音频时序设计；自回归图像生成成本更高，应作为独立的后期实验。** 不把四种输入输出能力放进一个联合训练任务后同时求收敛。

保留文本主干，分支新增输入适配器与语音输出模块。多模态阶段从已验收的 **文本 SFT** checkpoint 开始，不依赖完成 JustRL II、OPD 或 32K。

| 交付顺序 | 能力 | 默认目标 | 扩展条件 |
|---|---|---|---|
| M1 | 图像 → 文本 | 单图、固定分辨率的基础描述与问答 | 文本 SFT 与图文数据可用 |
| M2 | 语音 → 文本 | 短语音识别及简单语音问题回复 | audio projector 对齐通过 |
| M3 | 文本 → 语音 | 整句生成、一个固定音色的中英短句；先验收一种语言 | codec 重构及 Talker 训练通过 |
| M4 | 语音 / 图像 → 文本回复 → 语音 | 复用已验证的输入与输出分支，整轮交互 | M1–M3 的条件表示和回归通过 |
| M5，可选 | 文本 → 图像 | 256×256、有限领域、语义相关的生成研究 | 有合适 VQ 模型、成对数据和额外训练预算 |
| 后期实验 | 流式语音、更多音色、多图 / 高分辨率 | 单独设计训练与时延评测 | 整句方案已经可用 |

M1–M3 的先后可以按数据可得性调整。核心作品集可以止于文本模型 + 一个有证据的多模态分支，不以 any-to-any 的完整工业能力为验收要求。

### 0.1 原文中的关键错误

| 原判断 | 修订 |
|---|---|
| 语音输入直接进 Talker，不经过主干 | Talker 是输出分支；语音理解必须把语音特征经 projector 送入文本主干 |
| 冻结或极小 LR 主干，文本能力自然不退化 | 只有纯文本路径的权重、模板和输出空间完全不变时，才有相同 logits 的条件；联合微调、resize 和模式协议都可能改变行为 |
| 冻结编码器必须常驻 CPU、不能注册为 Module | 冻结与设备 / 注册方式是不同问题；可以注册、放 GPU，以推理效率和保存策略决定 |
| 冻结主干时整个 forward 用 no_grad | 输入 projector 需要通过主干得到梯度，不能把这条计算图切断 |
| 每个文本 hidden state 对应一个音频时间步 | 文本 token 与 codec 帧没有固定比率；必须使用独立语音时钟和明确条件机制 |
| delay pattern 让同一步各码本互相看到 | 同一解码步的 heads 仍同时预测；延迟让同一音频帧的较低层码本在更早解码步生成 |
| 多模态数据离线特征，同时随意 crop / 亮度增强 | 改像素后冻结编码器输出也会改变；缓存模式与在线增强必须二选一或缓存多个版本 |
| image 新行均值初始化保证不是噪声 | 只能缓和词表扩展的分布扰动，不能凭空学会生成图像；随机初始化也不必然导致梯度爆炸 |
| 256 个图像 codes 必然只占训练信号 1.5% | 比例取决于文本长度、任务混合和 mask，必须统计 |
| MiniCPM-o 2.6 用 CosyVoice2，并可作为图像生成先例 | 2.6 模型卡列的是 ChatTTS；该参考的图像能力是理解，不是本项目 VQ 图像生成的证据 |

## 1. 模块边界与 checkpoint

```text
文本 token ──────────────────────────────────────┐
图像 → 冻结视觉编码器 → image_projector ───────────┤
语音 → 冻结语音编码器 → audio_projector ───────────┤
                                                   ↓ inputs_embeds
                                            478M 文本主干
                                                   ↓
                                  ┌────────────────┴───────────────┐
                                  ↓                                ↓
                              text lm_head                 回复文本的 hidden states
                                  ↓                                ↓
                              文本回复                     条件投影 → 独立 Talker
                                                                   ↓
                                                        多码本 codes → codec decoder

可选图像输出：扩展主 lm_head → image codes → 冻结 VQ decoder
```

文本主干提供 `input_ids` / `inputs_embeds`、attention mask、position IDs、cache 和可选 hidden states 接口。多模态 wrapper 负责预处理、占位符映射、特征注入和任务输出，避免在每个 Block 中写模态分支。

保持文本 backbone 的结构与 key 兼容。多模态 checkpoint 新增 projector / Talker / codec manifest；图像输出还会 resize embedding。逐项验证允许的新增 keys、继承权重、词表和 tied storage，不使用不加解释的 `strict=False` 掩盖不匹配。

训练配置明确 `trainable_modules`、各模块 LR、初始化来源和 `encoder_mode: offline_features | online_frozen`。新增模块初始化后创建新阶段 optimizer，不把旧形状的 Adam 状态强行加载到 resize 后的参数上。

### 1.1 冻结模块的正确处理

编码器 `requires_grad_(False)`、`eval()`，提取特征时 `torch.no_grad()`。默认注册为正常 `nn.Module`，便于设备、dtype、部署与版本管理；冻结模块不进入 optimizer，导出训练 checkpoint 时只保存适配器 / 主干权重及编码器 repo、revision、processor hash，必要时提供完整部署打包。

注册 Module 不等于每步更新或 all-reduce 冻结参数。DDP 的初始化同步、buffer 同步和存盘范围按实现配置；无需用绕过注册来解决所有问题。大量离线预处理优先用 GPU 分批处理，计入算力预算，不要求在 CPU 上慢跑。

**冻结文本主干时仍允许输入梯度穿过主干。** 其参数不更新，但 `projector → inputs_embeds → backbone → loss` 必须保留计算图。只有不需要训练任何输入适配器、单独训练 Talker 并离线提取文本条件时，才可以对文本主干用 no_grad。

## 2. 图像输入：先固定分辨率

### 2.1 默认选型

| 项 | 起点与理由 |
|---|---|
| 编码器 | [SigLIP2 base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256)，仅使用 vision 分支，冻结 |
| 输入 | RGB，256×256；使用固定 revision 的官方 processor |
| 特征 | 8×8 = 64 个 patch；默认 hidden 768，以实际 encoder config / 输出核验 |
| projector | `768 → 1024 → 1024` 两层 MLP + GELU，约 1.84M 参数；可加输入 LayerNorm |
| 主干窗口 | 4K，先单图；不依赖 32K |
| 高分辨率 | 小字 / OCR 不足时再试更细 patch 或多 crop，单独记录 token 与计算成本 |

选择小编码器是资源和迭代效率的考虑，不是“冻结编码器大于主干就不可用”。固定 256 分辨率是首版工程取舍，动态分辨率也不是必然不可控。

不同编码器的 patch 公式不同。CLIP-ViT-L/14 的 224 / 336 可对应 256 / 576 patches，但不能把该数直接套到 SigLIP-SO400M 的其他 patch 配置。所有占位符数使用实际处理结果及 manifest，不依赖模糊的统一常量。

### 2.2 数据协议与特征注入

一张图使用一个显式 span：`<image_start>`、N 个 `<image_pad>`、`<image_end>`；控制符来自文本阶段保留的特殊 tokens，N 由特征数决定。多图时每个 span 关联独立 `media_id` 和顺序。

数据保存 `sample_id / media_id / modality / span / feature_length / feature_path`。校验必须逐样本、逐模态、逐 media span 进行；batch 总槽位数相等不足以证明没有样本串位。

```text
每个 media span：
  span 长度 == 对应 projected features 长度
  projected hidden_dim == backbone hidden_size
  feature_mask / attention_mask / position_ids / labels 对齐
  不允许缺图、重复注入、多余特征或截断 span
```

对不匹配的数据抛出包含 sample_id 的明确异常。训练缓存的校验不能只用可能被 Python 优化关闭的 `assert`。允许 batch 间长度不同及正常 padding，禁止用 padding 掩盖同一 span 的语义错配。

输入模态位置不计算输出文本 loss；assistant 回答正常计算。图像 span 不得被窗口截成半段，collator 需要知道“文本 tokens + 模态 slots”的真实总长度。训练和 prefill 用同一注入函数；decode 有 cache 后不反复编码 / 注入媒体。

本地 minimind-o 的 `inject_audio_features()` 使用 `min()` 截短，`count_vision_proj()` 可能在数量不符时跳过或改变长度。它们说明需要严格的数据契约，不能据此声称参考模型所有正常样本都一定存在丢信息的 bug。

### 2.3 图像训练

| 阶段 | 可训练模块 | LR 起点 | 规模与目的 |
|---|---|---|---|
| 对齐 | image_projector，主干冻结 | 1e-4–1e-3 | 5–20 万高质量 caption 对，先 1 epoch |
| 视觉指令 | projector + 主干 LoRA；必要时小 LR 全参 | projector 1e-4；LoRA 1e-4；全参主干 1e-6–1e-5 | 5–15 万独立问题 / 回答，再评估是否扩到百万级 |

对齐阶段纯文本样本不会给 image_projector 产生有效梯度，无需为“回放 20%”花同等训练预算；保持固定文本回归即可。主干解冻或加 LoRA 后，起点加入 **20–40% 的有效文本监督 tokens** 做回放，根据回归结果调整，不同时额外把所有图像样本乘 3–5 倍而不重算实际分布。

训练主干 LoRA 不会结构性保证原有能力不变。纯文本部署可以卸载独立 adapter 保留原权重；如果发布统一常开 adapter，则仍需要回归评测。

## 3. 语音输入：进入文本主干

### 3.1 默认路径与时间预算

```text
waveform → 重采样 / processor → 冻结 SenseVoice-Small encoder
         → 长度处理 / 可选下采样 → audio_projector
         → <audio_pad> spans → 文本主干 → 文本答案
```

以本地 minimind-o 使用的 **SenseVoice-Small** 为首个候选，输出 hidden 512，模型约 234M。具体 repo / revision、frontend、输出层和有效长度从实测确定，不能把输入 fbank 的长度自动当作 encoder 输出长度。英文覆盖不够时再比较 multilingual Whisper 等编码器。

projector 起点 `512 → 1024 → 1024`，约 1.58M 参数。先使用官方 processor 的有效 feature lengths，避免把 padding 当音频。长音频需要明确下采样 / 时间池化策略或完整音频切片，不能静默保留前 N 帧。

**编码器 feature rate 与输出 codec frame rate 是两个不同的量。** 输入音频不需要与 Talker 12.5Hz 强行对齐。先探查 5s / 10s / 30s 音频实际 slots、延迟与显存；首版限制约 10–20s 短音频，使总输入落在 4K 窗口内。

### 3.2 对齐和评测

先训练语音转录或有确定答案的音频任务，单训 audio_projector；再用音频指令数据训练回答，按需加入文本回放和主干 adapter。初轮 **10–100 小时**，不要直接承诺 5000 小时特征处理。

ASR transcript 可以作为监督输出或另一个 baseline 的输入，**不能在声称语音理解的评测中同时把答案转录偷偷喂给主干**。先建立外部 ASR → 文本模型的流水线作为可比较的系统 baseline；它有工程价值，与 learned audio projector 的模型实验分别报告。

指标包括中文 CER、英文 WER、音频问题成功率、噪声 / 静音 / 不同语速表现和音频编码时延。能转录不等于能完成音频语义任务；反之也不能仅看最终回答而跳过输入识别诊断。

## 4. 语音输出：独立 Talker、独立语音时钟

### 4.1 Codec 选型先于词表和 Talker

首选与本地参考一致的 [Mimi](https://huggingface.co/kyutai/mimi)，冻结 encoder / decoder。公开 config 和本地实现的核心规格：

| 项 | 首版选择 |
|---|---|
| 采样率 | 24kHz、单声道 |
| frame rate | 12.5Hz，约 80ms / frame |
| codebooks | 显式使用 8 层，确认 encoder / decoder 支持同一设置 |
| codebook size | **2048 / 层**，不是原设计固定的 1024 |
| 特殊 ID | 每层码本另加 BOS、EOS、PAD；原始 codes 不含这些控制值 |
| code 存储 | `[K, T]` uint16 + 有效长度 + codec manifest，labels 使用 int64 |

先做 **encode → decode 重构基线**，检查音质、CER / WER、长度与静音。codec 重构本身不合格时，不训练 Talker。可换其他 codec，但帧率、码本大小、K、延迟模式和已缓存 codes 全部需要重新核验，不能继续沿用 Mimi 的数字。

使用冻结的预训练 codec 是本项目的明确外部依赖；“从零训练”指文本主干与新增可训练模块，不指重训全部感知与声学组件。

### 4.2 第一版条件机制：完整文本 prefix

**先完成文本回复，再生成语音。** 不要求文本 token 数和音频帧数一致：

```text
文本主干生成最终回复（不朗读内部 think / 工具 JSON）
  → 使用相同文本及上下文取得回复 token 的 hidden states [S, 1024]
  → 条件 MLP [S, 512]，作为 Talker 的 prefix
  → 加固定说话人条件与语音 BOS
  → Talker 在独立帧时钟上生成多码本序列
```

首版可用主干最后一层 hidden states；中间层作为消融，固定层号并记录，不先承诺其一定更适合语音。训练使用标注文本的同一种 hidden 提取方式，推理使用实际生成文本的 hidden；统计两者差异并评测短句。完整文本条件给 Talker 提供发音信息，缓解“一个语义向量如何展开成任意长音频”的未定义问题。

初版对完整已生成回复做一次确定的主干 forward，提取实际回复 token 位置的 hidden states；不要直接把“用于预测下一个 token 的 hidden”错配为该生成 token 的输入表示。媒体特征和原始上下文保持一致，只选择应朗读的最终回复区间。

Talker prefix 的长度为文本 token 数 S；语音长度 T 由音频决定。它在声学区域自回归，prefix / 说话人 / PAD 不计语音目标 loss。该方案有意接受整句等待，作为验证声学建模的基线。

### 4.3 Talker 起始结构和参数量

```yaml
hidden_size: 512
num_hidden_layers: 8
num_attention_heads: 8
num_key_value_heads: 2
head_dim: 64
intermediate_size: 1792
qk_norm: true
rope_theta: 10000
max_position_embeddings: 4096         # 文本 prefix + 说话人 / BOS + delayed 音频序列
attention_bias: false
mlp_bias: false
num_codebooks: 8
codebook_size: 2048
speech_vocab_per_codebook: 2051         # codec codes + BOS/EOS/PAD
speech_embeddings: per_codebook
speech_heads: per_codebook
speech_tied_embeddings: false
text_condition: full_reply_prefix
```

每个解码步，将 8 层上一步的 delayed codes 分别 embedding 后求和并归一化，输入 Talker。8 个输出 heads 预测该步的各码本。prefix 使用投影特征，不走 speech embedding。

| 部分 | 参数量，约 |
|---|---:|
| 8 层 Transformer + 全部 norm | 27.27M |
| 8×2051×512 的 speech embeddings | 8.40M |
| 8 个 untied 输出头 | 8.40M |
| `1024→512→512` 条件 MLP，含 bias | 0.79M |
| **小计** | **44.86M** |

特殊说话人条件及后续模块另计，以独立参数枚举为准。原文的约 169M Talker 漏掉 untied 输出头及条件模块，也未确定 codec；本版以约 45M 起步，减少声学分支试验成本。若 capacity 不足，再根据训练 / 验证差距增加规模。

初轮限制目标语音不超过约 30s、文本 prefix 不超过 512 tokens；按 codec 有效帧和 delay 尾部计算总长度。超过限制的训练样本完整排除或在可靠的句段 / 时间对齐边界拆分，不截断文本和对应音频；推理超长回复先按完整句段合成。

### 4.4 Delay pattern 的精确定义

设原始 codes 为 `c[k,t]`，k 从 0 开始。先构造未加 BOS 的 delayed 目标矩阵：

```text
delayed[k,u] = c[k,u-k]，当 0 ≤ u-k < T
其余为无监督的 padding；每层 EOS 放在本层最后一个 code 之后
```

```text
解码步 u:   0    1    2    3    4
码本 0:    c00  c01  c02  c03  c04
码本 1:    PAD  c10  c11  c12  c13
码本 2:    PAD  PAD  c20  c21  c22
```

然后统一加 BOS、shift 一步构造 input / target。**delay 和 causal shift 是两件事**，不要两处各加一次 `+1`。对无效区、prefix、EOS 后的 padding 使用 loss mask；BOS / PAD 不参与有效区的输出采样。码本 k 在 `u < k` 的初始无效区由调度器直接填 PAD，不调用该 head 的采样。

同一音频帧 t 的码本 k 在解码步 `t+k` 生成，此时较低码本对应帧的 codes 已在更早步生成并可进入上下文。同一个 u 内的 8 个 heads 仍同时预测，不能说它们互相看到了当步的新输出。

stop 规则：由 codebook 0 的 EOS 确定最终帧数 T；其他码本在其有效区屏蔽 EOS，在延迟对应的 `u=T+k` 由调度器结束。codebook 0 结束后仍补齐其他码本尚未生成的尾部，全部结束后再还原成 `[K,T]`。若数据 EOS 布局或最终帧数不符，报告无效序列；设最大 frame 数防止失控，禁止不检查长度直接交给 decoder。

训练、采样、flush、de-delay 使用同一 pattern 定义。往返相等只验证布局，还必须测试 causal shift、future-code 泄漏与自回归条件；能够重建布局不代表模型学到了可生成的音频。

### 4.5 Talker 训练

| 阶段 | 可训练模块 | 起点 |
|---|---|---|
| 小样本验证 | Talker + 条件 MLP，主干冻结 | 少量短句和固定音色；检查 labels、音频重构与自由生成 |
| T2A | 同上 | 100–300 小时目标语音，先一种语言，LR 1e-4–3e-4，warmup 1–3%，先 1 epoch |
| 扩充语言 / 音色 | 同上，按需加入明确的 speaker 条件 | 根据 dev 发音与数据覆盖扩大 |
| 联合微调，可选 | 主干 adapter / 小 LR + Talker | 只有冻结条件已收敛且联合训练有可解释收益才开启 |

loss 分别报告每个码本的有效 CE、帧数、EOS 和重构指标，避免单个低层码本或 padding 主导总 loss。teacher-forcing loss 下降后必须跑自由生成，检查重复音节、漏字、尾音、音频长度和 speaker 稳定性。

本地 minimind-o 的英文 mini T2A 约 470 小时，能作为流程参考；不能据此承诺中文也在相同成本内可用。其 full T2A 约 1636 小时，提供了更现实的扩充规模锚点。

## 5. 图像输出：独立的高风险研究分支

### 5.1 条件和词表扩展

候选采用 `vqgan-f16-16384` 一类冻结 VQ tokenizer / decoder。**先确认实际模型来源、权重许可、码本大小、下采样倍率和重构质量，再决定 resize**。首版目标 256×256、f16 潜空间 16×16、每张图 256 codes；不同 VQ 模型不自动满足这些数值。

若确认码本 16384，映射 `image_token_id = 49152 + codec_id`，`codec_id ∈ [0,16384)`。扩展主 embedding / tied lm_head 到 65536；新增权重 **16,777,216**，主干总计 **494,998,528**，约 495M，尚不含任何其他模态模块。

VQ 码本 16384 只是离散条目数，不是“128×128 的空间码本”；空间 grid 由图像分辨率和 encoder 下采样确定。图像输入的连续视觉特征与输出的 VQ codes 是两种表示，本项目先分别建模。

代码定义 `text_vocab_size`、`model_vocab_size` 和 image-code manifest。image codes 作为整数映射，不参与 BPE merges；若为 HF 导出登记对应 token 字符串，也要验证普通文本编码与老 ID 未变。新增词表行数、codec codebook、mode mask 和 tied storage 一起校验。

### 5.2 初始化、模式约束和能力保护

新行可用旧 embedding 的均值加小噪声或有界分布初始化，并在 pilot 比较；全均值不是学会图像的保证，完全相同的初始化也需要检查早期对称性。加载后必须验证 embed / head 仍共享权重，重新创建新行的优化器状态。

使用显式输出模式：

- **文本模式**：屏蔽新增 image-code logits，保留原文本允许集、模板和采样规则。只保持旧 ID 不变，不会自动保住 softmax 概率。
- **图像模式**：prompt / image-start 之后，仅采样对应 image-code 区间；固定 grid 满 256 codes 后补 image-end，不把图像 ID 交给文本 tokenizer 解码。
- 图像 codes 参与 CE，控制符策略与推理一致；训练和推理使用相同的 allowed-token mask，避免训练 full softmax、推理另一分布而没有说明。

词表扩展、主干微调均可能影响文本，因此图像生成用独立 checkpoint / adapter，先验收回归再考虑合并。新增行 + 冻结主干只能建立低成本 baseline；如果主干不会根据 caption 建模二维纹理，训练新 embedding 并不一定足够。

### 5.3 数据与目标

初轮选择有明确 caption 的有限领域，**10–30 万对**，先离线编码并检查重构 / 相关性；百万级广域生成需要额外预算。采用 caption→codes 自回归训练，语义理解和图像生成两个方向使用各自的 labels 与任务采样。

统计 text / image 的有效目标 tokens 及按任务的 loss；若图像梯度过少，显式调整任务采样或 loss 权重，并报告最终曝光比例。不能同时用高过采样和高 loss 权重却不说明实际放大倍数。

指标先看 VQ 重构、有效 code 比例、codebook 使用率、caption 条件相关性和成对 prompt 人工评测。FID 只在足够且可比较的样本上报告；有限数据下不以“只要不是噪声”作为全部验收。冻结 VQ decoder 本身不能保证从零自回归 prior 学到开放域图像生成。

## 6. 数据、缓存与质量

### 6.1 来源与阶段规模

| 任务 | 候选来源 | 首轮规模 | 核验事项 |
|---|---|---|---|
| 图像对齐 / 指令 | minimind-o README 指向的 MiniMind-V 来源；[LLaVA 预训练数据](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) / [指令数据](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) 的可用子集 | caption 5–20 万对，指令 5–15 万条 | 图片实际可得、原始许可、问题依赖图像、文字描述准确 |
| 中文语音输入 | AISHELL-1；目标指令数据的合法子集 | 10–100 小时 | transcript / waveform 对应，采样率与说话人划分 |
| 英文 / 多语言语音输入 | [FLEURS](https://huggingface.co/datasets/google/fleurs)，选择相应语言与 splits；可用语音指令子集 | 先短音频与小批，再按任务扩充 | 音频许可、语言标记、转录与实际内容 |
| Talker | [minimind-o dataset](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset) 的 T2A；中文可探查 AISHELL-3 等 TTS 语料 | 100–300 小时起，一种语言 / 音色先验收 | codec 版本、音频 / 文本对齐、发音质量、说话人条件和使用范围 |
| 音频对话 | 上述数据中的 A2A、VoiceAssistant-400K 等参考来源 | 先 1–5 万条短轨迹 | 是否合成、标注一致、问题音频与答案语音分开 |
| 图像生成 | 已核验许可、caption 相关性与 VQ 重构的独立图文子集 | 10–30 万对 | 训练 / 验证图像隔离，来源和 caption 多样性 |

上述是**候选和抽样目标，不是已下载、已清洗的数据**。先检查原始授权及数据质量。不要依赖 LAION / CC12M 的 URL 总条数作为可训练图片数；失效链接、下载时间和网页 caption 噪声需单列预算。

图像按原图及近重复簇划分，语音按说话人 / 录音来源划分；同一个源内容的合成版本与原始版本不得跨 train-dev-test。多 caption 对同一张图是合法数据形式，不应因为描述不同就全部去掉；去重目的是控制曝光和泄漏。

强模型改写 caption 是可选质量实验，不默认必然优于人工 / 原始标注；CLIP score、OCR、统计筛选和人工抽查可以发现语义错配，但都不保证彻底排除。自动质量诊断与人工核查结合，不声称“语义错配无法用代码检测”。

### 6.2 离线与在线两种模式

| 模式 | 适用场景 | 契约 |
|---|---|---|
| 离线冻结特征 | 快速 projector 迭代、processor 固定 | encoder / processor / layer / dtype / crop 哈希一致，不能随意更改原图增强 |
| 在线冻结编码器 | 对增强 / 分辨率进行实验 | GPU 提取特征，no_grad 编码器；projector 与主干梯度路径保留，重新测吞吐 |
| 离线 codec codes | 声学 / VQ 生成 | waveform / 图像处理、codec revision、K / C / rate 和有效长度固定 |

索引统一组织 sample、media、文本、slots、codes、有效 mask、数据来源和 split。不要给百万样本各放一个小 `.npy`；采用带 offsets 的分块数组、Arrow / Parquet 或其他能顺序读取的 shard，并记录 checksum。缓存失效时显式重建，避免不同版本特征混训。

合理的文本 / 问题位置变化要保持媒体引用与任务语义，不能为了“位置随机化”把 user 图像放到 assistant 答案后面而造成泄漏。预计算特征下，像素级增强必须提前缓存多个版本，或切换为在线特征模式。

### 6.3 磁盘预算必须计输入特征

以 float16 特征 / uint16 codes、十进制 GB 计算：

| 缓存 | 估算 |
|---|---:|
| 100 万图，64×768×2 bytes / 图 | **98.3GB** |
| 100 万图，256×768×2 bytes / 图 | **393.2GB** |
| 5000 小时 Mimi codes，8×12.5Hz×2 bytes | **3.6GB** |
| 1000 小时音频输入特征，假设 50Hz×512×2 bytes | **184.3GB** |
| 24kHz 单声道 PCM16，每小时 | **172.8MB** |

50Hz 的输入 feature rate 是预算示例，SenseVoice 实际 rate 由 probe 结果替换。原文只算语音输出 codes、遗漏输入 encoder 特征，因此“只有图像缓存会撑爆 SSD”不成立。

复用文本设计的远端 shard / 有限缓存机制，保留当前与预取 shard、文本回放数据及 checkpoint 原子写空间。特征生成也可在训练服务器 GPU 上分批完成；是否另租机器取决于吞吐和费用，冻结与 CPU 常驻无必然关系。

## 7. 统一训练约定与阶段控制

### 7.1 每次只新增可诊断的能力

对输入模态先单训 projector，若验证集依然停滞，再考虑主干 adapter / 小 LR 全参。对输出语音先只训 Talker 和条件投影；输入与输出各自成立后再组合。图像生成独立分支，不混入最初的图像理解对齐。

单训新模块是稳定的起点，**不是所有多模态方法必须先冻结，否则一定遗忘的定律**。根据任务和数据可以选择联合训练，但必须有冻结 baseline 和回归证据。

### 7.2 文本与模态损失

- 文本 CE、音频各码本 CE、图像 CE 分开汇总，统一用有效 tokens / frames 定义分母。
- 各任务 loss 权重和采样比例共同决定梯度贡献；日志报告实际曝光，不只保存 YAML 上的样本比例。
- 混合 batch 中缺少某模块输入时，正确处理 DDP unused parameters；不能通过给假图 / 假音频贴上合法 label 制造虚假监督。
- 模态 slot 的 PAD、文本 PAD、speech PAD、ignore label 是不同概念，分别编码。
- label / span 校验在离线准备及训练 collator 两处执行，错误样本计数进入 manifest。

多模态不需要跨宽度 μP 搜索，默认采用文本 checkpoint 的原参数化及模块 LR。若主干本来经过 μP 训练，仍应保留其前向缩放与权重语义；不能在微调时因为“不再用 μP 搜超参”而把主干改成另一套前向。

### 7.3 冻结不代表不用回归

每个发布候选同时跑固定文本集和相应模态集。建议初始门槛：文本平均准确率绝对下降不超过 2 个百分点、dev NLL 相对上升不超过 3%，并报告样本数与置信区间；小评测集无法分辨时增加样本。阈值是项目约定，正式长跑前固定，不能训练后追着结果改。

结构 / mask 改变时先确保 NLL 评测仍比较相同文本允许集。冻结所有文本路径、未 resize、未改模板的 projector-only 模型，应能在无媒体输入下复现原模型 logits；解冻、LoRA 或图像词表扩展则用实际行为和数值回归评估。

## 8. 推理、流式与部署

### 8.1 首版整轮交互

```text
图像 / 语音预处理 → 注入特征 → 文本回复
                                   ↓
                           完整文本条件 → Talker
                                   ↓
                           de-delay → codec → 整句音频
```

输入 feature cache、文本 KV cache、Talker KV cache 和 codec decoder state 分开管理，reset、batch 变化和会话结束时显式清理。缺少必要模态资源时报告清楚的错误，不静默把问题降级成纯文本猜测。

### 8.2 流式语音是第二个设计任务

整句 prefix 方案不能一边还没生成完整文本、一边声称 Talker 已在完整文本条件下流式发声。可先做**句段级**输出：主干生成完整句段后发给 Talker；进一步降低时延需要文本 chunk 可见性、单调对齐 / 调度和相应训练数据，重新定义条件机制。

Mimi 12.5Hz 下 4–8 帧是 **320–640ms**；8 路 delay 的完整帧组装还有最多 7 帧、约 **560ms** 的码本延迟。必须区分模型首帧、完整可解码 chunk 和用户首个可听样本，不把所有延迟相加或忽略而不说明调度重叠。

只有 decoder 确认支持状态化增量解码时才复用其流式状态；独立解码短 chunk 再拼接会产生边界问题。TTFA 计入输入编码、文本条件等待、Talker prefill / decode、delay、codec 和播放缓冲。首个文本 token 的 TTFT 不能替代首个可听音频的 TTFA。

本轮不要求全双工、实时打断和自然语音克隆。整轮 demo 验收后，再以真实延迟和音质决定是否增加 VAD、流式编码器、声音控制及会话栈。

### 8.3 部署与模型规模披露

CLI → Web demo，输入模态、文本输出、整句语音播放明确展示。图像输出采用独立开关 / checkpoint，模式由 API 或显式控制符指定，并使用相应 vocab mask。

分别公布：文本主干参数、新增可训练参数、冻结视觉 / 音频编码器与 codec 参数、运行时总加载和显存。不能把 478M 主干加上所有外部模型后仍称为“运行时只有 0.5B”。纯文本模式可以按需不加载外部模块。

## 9. 验收和必要验证

| 阶段 | 功能与证据 | 必需对照 / 失败诊断 |
|---|---|---|
| M1 图像理解 | 物体 / 颜色 / 数量问答，固定独立测试集；多图只在声明支持后测试 | 换图、错图、去图对照，排除只靠文本先验；无图不能仅因一个样例错就断言注入失效 |
| M2 语音输入 | CER / WER、简单问题成功率，噪声 / 静音测试，编码时延 | 外部 ASR 流水线、真实 audio-only 任务；确认没有答案 transcript 泄漏 |
| M3 语音输出 | 自由生成短句可听懂，ASR CER / WER、漏字 / 重复、音频长度和固定音色稳定性 | codec 重构、teacher-forcing 与自由生成差距；人工抽查自然度，不只用 ASR 分数 |
| M4 组合 | 独立音频 / 图像问题得到正确文字与可听回复，测整轮时间 | 输入与输出模块逐个关闭排查；改变组合条件需重新评测 Talker |
| M5 图像生成 | 有效 grid / codes、相关性、codebook 使用与独立 prompt 样例 | VQ 重构、无 caption / 换 caption 对照；不把生成噪声只归因于 embedding 初始化 |

所有阶段同时提交文本回归表、实际数据曝光、训练曲线、资源预算和 checkpoint manifest。对 held-out 数据先固定约 100–300 个代表性样本，图像换图和音频发音检查另设诊断集；随着能力扩大再引入公开 benchmark，说明低分和样本限制。

必要实现验证：

- 逐 media span 的 slots、特征数、mask 与位置对齐；缺资源、错长度、错误 batch 映射必须失败。
- 冻结主干时 projector 梯度非零，主干参数不变；在线 / 离线相同 processor 特征一致到合理容差。
- 训练 / prefill 使用同一注入协议，cache decode 不重复注入；不同长度 batch 不串位。
- codec code 范围、采样率、K、有效帧数与 reconstruction 正确。
- delay / de-delay 往返、BOS shift、EOS flush、future-code 不可见性及多码本 padding loss 正确。
- resize 保持旧 ID / 旧行、tied storage 与模式 mask；新增 optimizer 状态形状正确。
- 文本 prefix 条件训练 / 推理一致；不能向 Talker 提供训练标签中的未来音频 codes。
- 多模态保存 / 加载后 encoder、processor、codec、模板和实际权重版本一致。

本地 smoke data 用 16–32 个短样本，先验证梯度和布局，再尝试记忆；音频要包含原始 waveform 和 codec 重构。loss 接近零不保证推理链路、声学自然度或泛化，因此必须额外跑自由生成和独立评测。

## 10. 执行前的决定与停机条件

1. 从文本 SFT 固定版本开始，先验证图像与音频 processor、codec 重构和数据许可。
2. 完成一个输入模态的小规模对齐与独立评测，检查 projector 梯度和文本回归。
3. 完成一个语言 / 固定音色的整句 Talker，再组合输入与输出；新增能力按数据和收益决定。
4. 只在长句发音、回归和磁盘流水稳定后扩充规模；图像生成与流式语音各自单列预算。

尚待实测：编码器的精确输出层 / rate、短样本质量、中文 T2A 来源与音色条件、Talker 条件层、VQ 权重与许可、各阶段 LR / 任务混合、端到端吞吐与 TTFA。先以本版默认值探查，不把未知数据源写成“已具备百万级训练对”。

错位、丢特征、codec 重构失真或文本明显退化时停止扩容；修复并复验后再继续。M5 若仅能记忆训练图像，保留研究记录，不并入已获得的开放域生成能力。

## 11. 核验来源与参考边界

| 来源 | 本次采用与纠正 |
|---|---|
| 本地 [minimind-o 模型](../minimind-o/model/model_omni.py)、[数据](../minimind-o/dataset/omni_dataset.py)、[README](../minimind-o/README.md) | commit `f900448c608318c53314ebf8a947ab05cd8c038e`。可参考 projector、输入注入、独立 Talker、delay 和数据组织；本项目改用严格 span 契约和先整句条件 |
| [Mimi config](https://huggingface.co/kyutai/mimi/blob/main/config.json) | 24kHz、12.5Hz、2048 码本；8 codebooks 的使用与数据参考来自本地 minimind-o，真正缓存前锁定 codec revision 并实测 |
| [SigLIP2 base config](https://huggingface.co/google/siglip2-base-patch32-256/blob/main/config.json) | 256 输入、patch32，对应 64 patches；以 vision 子模块实际 config / processor 核验维度 |
| [MiniCPM-o 2.6 模型卡](https://huggingface.co/openbmb/MiniCPM-o-2_6) | 列出 SigLip-400M、Whisper-medium-300M、ChatTTS-200M、Qwen2.5-7B，合计约 8B。参考其图像 / 音频理解和语音交互，不能把它写成 CosyVoice2、VQ 图像生成或本项目 delay 方案的直接证明 |
| [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | Thinker / Talker 的分工思路；不声称自建约 45M Talker 就能复现其流式对齐与效果 |
| [LLaVA](https://github.com/haotian-liu/LLaVA) | 固定视觉编码器、projector 与视觉指令训练的工程参考；具体数据许可和图像可得性单独核实 |

本次完成的是设计审查与重写、公开资料和本地实现核验、参数及存储算术复核；未训练多模态模型，也未验证候选数据的全部样本。实施记录需要逐步替换本文的预算示例和假设。

2026-10-05 补查 Hugging Face 候选页面时遇到网络连接错误，未完成当前可访问性复核；上文保留此前已读资料的版本证据，候选数据仍按实际使用前的探查结果决定。
