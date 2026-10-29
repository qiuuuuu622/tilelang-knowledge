# 模式来源与证据边界

核查日期：2026-10-29。这里登记正式引用的 15 条已合并 GPU PR、3 个固定源码仓库及社区补充资料。只读检索与静态核查；**没有本地 GPU 实验，也没有独立复跑上游 benchmark**。

- GitHub API 已核实各 PR 的 merged 状态与合并 SHA，并核实下列固定版本源码路径存在。
- 合并源码用于判断最终行为；PR 正文/讨论用于了解动机、作者测量与撤回尝试。正文可能滞后于最终 diff，讨论和图表也可能事后修改，不能视为不可变的原始实验产物。
- 这些是历史版本证据，不代表现在 main 的全部行为。各来源依赖不同，不能拼成一个默认兼容的运行环境。
- CUDA/Triton/HIP 的可迁移机制与 TileLang 原生实现分别标注。同源移植不是独立复现；源码存在测试函数不等于本地测试通过。
- 本轮不复制上游完整代码、图片或原始日志；解释文字与短事实摘要不改变上游许可。SGLang、vLLM、OneFlow、CUTLASS 使用 Apache-2.0，BBuf 仓库的代码/转载材料需逐文件核对许可，勿据此复制再分发。已有代码案例的许可见 [kernels 来源登记](../kernels/SOURCES.md)。

## 正式 PR 证据

### S01

- **[sgl-project/sglang #31324：[AMD] [GLM5] Skip DSA decode indexer when kv_len <= index_topk (dense k-only fast path)](https://github.com/sgl-project/sglang/pull/31324)**；合并日期（UTC）2026-08-16。
- 合并 commit：[`f7cb328eb74929e70f4f7675269abebb5da05453`](https://github.com/sgl-project/sglang/commit/f7cb328eb74929e70f4f7675269abebb5da05453)。
- 固定代码：[`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`](https://github.com/sgl-project/sglang/blob/f7cb328eb74929e70f4f7675269abebb5da05453/python/sglang/srt/layers/attention/dsa/dsa_indexer.py)。
- 阅读结论与边界：TileLang DSA 调用链。核对 k-only 分支、K cache 更新与混合长度回退。性能来自作者 MI355X/GLM-5.2 TP4 的指定短上下文测试；双 graph 的成本不省略。

### S02

- **[sgl-project/sglang #43463：[QSA] Bound the decode indexer's work by the context length](https://github.com/sgl-project/sglang/pull/43463)**；合并日期（UTC）2026-10-12。
- 合并 commit：[`23a1f53fc3a0e7f5fc94c4a4abd4eb9bcf9c6042`](https://github.com/sgl-project/sglang/commit/23a1f53fc3a0e7f5fc94c4a4abd4eb9bcf9c6042)。
- 固定代码：[`python/sglang/srt/layers/attention/qsa/mqa.py`](https://github.com/sgl-project/sglang/blob/23a1f53fc3a0e7f5fc94c4a4abd4eb9bcf9c6042/python/sglang/srt/layers/attention/qsa/mqa.py)；[`python/sglang/srt/layers/attention/qsa/fused_kv.py`](https://github.com/sgl-project/sglang/blob/23a1f53fc3a0e7f5fc94c4a4abd4eb9bcf9c6042/python/sglang/srt/layers/attention/qsa/fused_kv.py)。
- 阅读结论与边界：TileLang scoring + Triton KV prepare。核对有效长度、有界 CTA、跳过评分及消费者读取边界；B200 的 step 收益为组合改动，不拆成单项因果。

### S03

- **[vllm-project/vllm #59464：[GLM5.3 Perf] Reuse sparse MLA index conversion across layers, 3.5~3.9x kernel performance improvement](https://github.com/vllm-project/vllm/pull/59464)**；合并日期（UTC）2026-10-02。
- 合并 commit：[`4e4d75c4594b1072729ffe335c9bcea601749f71`](https://github.com/vllm-project/vllm/commit/4e4d75c4594b1072729ffe335c9bcea601749f71)。
- 固定代码：[`vllm/v1/attention/backends/mla/flashattn_mla_sparse.py`](https://github.com/vllm-project/vllm/blob/4e4d75c4594b1072729ffe335c9bcea601749f71/vllm/v1/attention/backends/mla/flashattn_mla_sparse.py)。
- 阅读结论与边界：GPU sparse MLA 索引复用。核对 index group、多层、全 decode 的条件。PR 的 3.5–3.9× 是转换微基准，模式未引用为 attention/模型加速。

### S04

- **[sgl-project/sglang #41030：[AMD] Eliminate remaining gfx95 FP8 scale relayout copies](https://github.com/sgl-project/sglang/pull/41030)**；合并日期（UTC）2026-10-09。
- 合并 commit：[`d5c3ae975282eaa312b0e033517243e2b3f3ae4e`](https://github.com/sgl-project/sglang/commit/d5c3ae975282eaa312b0e033517243e2b3f3ae4e)。
- 固定代码：[`python/sglang/srt/layers/layer_boundary/residual/add_norm.py`](https://github.com/sgl-project/sglang/blob/d5c3ae975282eaa312b0e033517243e2b3f3ae4e/python/sglang/srt/layers/layer_boundary/residual/add_norm.py)。
- 阅读结论与边界：ROCm FP8 producer/consumer 接线。核对 gfx95、M≥2 gate、转置 scale view 和 M=1 回退；不是 TileLang 专属实现。测量是作者指定 MI355X/GLM-5.3 配置。

### S05

- **[vllm-project/vllm #42774：[Perf] Padded nvfp4 quant kernel to remove additional copy, 2.4%~5.7% e2e performance improvement](https://github.com/vllm-project/vllm/pull/42774)**；合并日期（UTC）2026-05-18。
- 合并 commit：[`37ece593c105b5bb818aa94885617b863d390d7f`](https://github.com/vllm-project/vllm/commit/37ece593c105b5bb818aa94885617b863d390d7f)。
- 固定代码：[`csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/37ece593c105b5bb818aa94885617b863d390d7f/csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu)。
- 阅读结论与边界：CUDA NVFP4 量化直接写 padded 输出。核对物理宽度、scale 与 wrapper 的后端限制。原文部分速度表的计时含 quant+GEMM，且未明确具体 GPU 型号，模式只使用代码机制，不引用该数字。

### S06

- **[vllm-project/vllm #63382：[Bugfix][NVFP4] Fix NaN logits and garbage output from uninitialized fused act+quant scales](https://github.com/vllm-project/vllm/pull/63382)**；合并日期（UTC）2026-10-29。
- 合并 commit：[`b42f25b00c3d13b565ec60e81928cd21efefce32`](https://github.com/vllm-project/vllm/commit/b42f25b00c3d13b565ec60e81928cd21efefce32)。
- 固定代码：[`csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu`](https://github.com/vllm-project/vllm/blob/b42f25b00c3d13b565ec60e81928cd21efefce32/csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu)。
- 阅读结论与边界：CUDA fused SiLU+NVFP4 修复。padding scale 的物理读取导致 NaN；kernel 内补零保留 empty 分配。作者验证为 sm120 RTX PRO 4500，不能把该错误推断为所有量化实现都有问题。

### S07

- **[sgl-project/sglang #18488：Tilelang sparse decode fwd for dsv32 mi355](https://github.com/sgl-project/sglang/pull/18488)**；合并日期（UTC）2026-02-10。
- 合并 commit：[`4262f5259b94c5c08779efff5017c0a6235bcb5f`](https://github.com/sgl-project/sglang/commit/4262f5259b94c5c08779efff5017c0a6235bcb5f)。
- 固定代码：[`python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`](https://github.com/sgl-project/sglang/blob/4262f5259b94c5c08779efff5017c0a6235bcb5f/python/sglang/srt/layers/attention/nsa/tilelang_kernel.py)。
- 阅读结论与边界：TileLang sparse decode 的 partial/combine 与 query≤64 分派。作者 MI355 的小/大 query 对照同时保留；kernel 与端到端两种口径不混用。

### S08

- **[sgl-project/sglang #32755： [Perf] Occupancy tuning for DSA indexer fp8-quant Q kernel](https://github.com/sgl-project/sglang/pull/32755)**；合并日期（UTC）2026-08-13。
- 合并 commit：[`dbebc1deb42b00befa3d0de67265d7003994c1ad`](https://github.com/sgl-project/sglang/commit/dbebc1deb42b00befa3d0de67265d7003994c1ad)。
- 固定代码：[`python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh`](https://github.com/sgl-project/sglang/blob/dbebc1deb42b00befa3d0de67265d7003994c1ad/python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh)。
- 阅读结论与边界：CUDA indexer 线程块调优。已核对最终生产 diff 仅保留 128→256 与注释；不能照抄 PR 首段的 persistent、lane-0 写入或可调模板描述。消融与最终收尾见下方讨论。

消融：[persistent 与仅增大 CTA 的对照](https://github.com/sgl-project/sglang/pull/32755#issuecomment-5139475942)；最终改动：[移除其他改动后的审查回复](https://github.com/sgl-project/sglang/pull/32755#discussion_r3719663666)。

### S09

- **[vllm-project/vllm #56478：[Kernel][Perf][Quantization] Fix odd-row performance cliff in per-token-group quantization](https://github.com/vllm-project/vllm/pull/56478)**；合并日期（UTC）2026-09-13。
- 合并 commit：[`c2ea9f1a5f10caec36eac91c461fcec6564f5394`](https://github.com/vllm-project/vllm/commit/c2ea9f1a5f10caec36eac91c461fcec6564f5394)。
- 固定代码：[`csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`](https://github.com/vllm-project/vllm/blob/c2ea9f1a5f10caec36eac91c461fcec6564f5394/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu)。
- 阅读结论与边界：CUDA/ROCm 共用量化源文件，所引用数字来自 H100。核对奇数行启动策略、有效 group mask、全体 barrier/PDL 参与；没有为输入分配 padded 副本。

### S10

- **[vllm-project/vllm #56633：[Perf][DSv4.1] Fold the mHC post block into the delayed pre projection](https://github.com/vllm-project/vllm/pull/56633)**；合并日期（UTC）2026-09-14。
- 合并 commit：[`5372e72a9884c141874465f0eb8054a4a81da6bb`](https://github.com/vllm-project/vllm/commit/5372e72a9884c141874465f0eb8054a4a81da6bb)。
- 固定代码：[`vllm/model_executor/kernels/mhc/tilelang_kernels.py`](https://github.com/vllm-project/vllm/blob/5372e72a9884c141874465f0eb8054a4a81da6bb/vllm/model_executor/kernels/mhc/tilelang_kernels.py)；[`vllm/model_executor/kernels/mhc/tilelang.py`](https://github.com/vllm-project/vllm/blob/5372e72a9884c141874465f0eb8054a4a81da6bb/vllm/model_executor/kernels/mhc/tilelang.py)；[`vllm/models/deepseek_v41/nvidia/model.py`](https://github.com/vllm-project/vllm/blob/5372e72a9884c141874465f0eb8054a4a81da6bb/vllm/models/deepseek_v41/nvidia/model.py)；[`vllm/model_executor/kernels/mhc/warmup.py`](https://github.com/vllm-project/vllm/blob/5372e72a9884c141874465f0eb8054a4a81da6bb/vllm/model_executor/kernels/mhc/warmup.py)。
- 阅读结论与边界：TileLang delayed mHC post/pre 融合。核对 wrapper、tile/shape gate、Engram 分支及 warmup。采用调优后的未融合基线；GB300、hc=4、hidden=5120 的 crossover 不推广到所有 hidden/GPU。

### S11

- **[vllm-project/vllm #57984：[Bugfix][Kernel] Skip the fused silu-mul block-quant fast path when a swiglu clamp is set](https://github.com/vllm-project/vllm/pull/57984)**；合并日期（UTC）2026-09-21。
- 合并 commit：[`8bf051cd51adff79e1bc94e910adaea543aaa2e1`](https://github.com/vllm-project/vllm/commit/8bf051cd51adff79e1bc94e910adaea543aaa2e1)。
- 固定代码：[`vllm/model_executor/layers/fused_moe/experts/triton_moe.py`](https://github.com/vllm-project/vllm/blob/8bf051cd51adff79e1bc94e910adaea543aaa2e1/vllm/model_executor/layers/fused_moe/experts/triton_moe.py)。
- 阅读结论与边界：Triton MoE activation/quant 快路径 guard 修复。只作为 clamp 语义边界的反例，不作 TileLang 性能证据。

### S12

- **[sgl-project/sglang #21654：Fused_qknorm_rope kernel optimization: up to 2.4× faster](https://github.com/sgl-project/sglang/pull/21654)**；合并日期（UTC）2026-04-01。
- 合并 commit：[`5bbf347bb31d64807c00d4401772f0f69b04a51f`](https://github.com/sgl-project/sglang/commit/5bbf347bb31d64807c00d4401772f0f69b04a51f)。
- 固定代码：[`python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`](https://github.com/sgl-project/sglang/blob/5bbf347bb31d64807c00d4401772f0f69b04a51f/python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh)；[`python/sglang/jit_kernel/fused_qknorm_rope.py`](https://github.com/sgl-project/sglang/blob/5bbf347bb31d64807c00d4401772f0f69b04a51f/python/sglang/jit_kernel/fused_qknorm_rope.py)。
- 阅读结论与边界：CUDA QKNorm+RoPE。核对 interleaved 分支、频率递推、向量加载、YaRN JIT 参数；B200 的多个优化合并测量不能分别归因。

### S13

- **[vllm-project/vllm #37376： fused qknorm+rope kernel optimization for SM9.0](https://github.com/vllm-project/vllm/pull/37376)**；合并日期（UTC）2026-04-13。
- 合并 commit：[`4beeb0689cb6d86c5dbc6a8688a468892196b540`](https://github.com/vllm-project/vllm/commit/4beeb0689cb6d86c5dbc6a8688a468892196b540)。
- 固定代码：[`csrc/fused_qknorm_rope_kernel.cu`](https://github.com/vllm-project/vllm/blob/4beeb0689cb6d86c5dbc6a8688a468892196b540/csrc/fused_qknorm_rope_kernel.cu)。
- 阅读结论与边界：CUDA fused QKNorm+RoPE。核对多 head/warp 实现与按负载选择；作者 H100 结果不直接推广到 B200/AMD 或 TileLang。

### S14

- **[vllm-project/vllm #44692：[Bugfix][Kernel] Fix mHC fused-RMSNorm big-fuse miscompile for hidden_size != 4096](https://github.com/vllm-project/vllm/pull/44692)**；合并日期（UTC）2026-06-06。
- 合并 commit：[`ec0a31d4aa019567395d9e7cce67d1e6ac56974a`](https://github.com/vllm-project/vllm/commit/ec0a31d4aa019567395d9e7cce67d1e6ac56974a)。
- 固定代码：[`vllm/model_executor/kernels/mhc/tilelang_kernels.py`](https://github.com/vllm-project/vllm/blob/ec0a31d4aa019567395d9e7cce67d1e6ac56974a/vllm/model_executor/kernels/mhc/tilelang_kernels.py)。
- 阅读结论与边界：TileLang fused RMSNorm 流水修复。最终 diff 为 num_stages 3→2；错误涉及特定依赖与 shared buffer 组合，不能写成 stage=3 普遍不正确。验证为作者 GB200/CUDA 13 报告。

### S15

- **[vllm-project/vllm #58419：[ROCm][Bugfix] Fix TileLang mHC fused RMSNorm on 64-wide wavefronts](https://github.com/vllm-project/vllm/pull/58419)**；合并日期（UTC）2026-09-23。
- 合并 commit：[`5fcc6e7c7380c67ab1e1305ed2197b127f0535e5`](https://github.com/vllm-project/vllm/commit/5fcc6e7c7380c67ab1e1305ed2197b127f0535e5)。
- 固定代码：[`vllm/model_executor/kernels/mhc/tilelang_kernels.py`](https://github.com/vllm-project/vllm/blob/5fcc6e7c7380c67ab1e1305ed2197b127f0535e5/vllm/model_executor/kernels/mhc/tilelang_kernels.py)。
- 阅读结论与边界：TileLang HIP wave64 归约分组修复。核对 AMD 线程角色重排、CUDA 保持原路径；作者 MI355X 测试从 27 failed 变为通过。它是分组正确性约束，不是性能收益。

## 源码与机制资料

### S16

**OneFlow Softmax：缓存与重读的分段策略。**

- [知乎原文](https://zhuanlan.zhihu.com/p/341059988)、[作者官网版本](https://oneflow.ai/a/share/jishuboke/54.html)。
- [固定源码 softmax.cuh](https://github.com/Oneflow-Inc/oneflow/blob/25c8978c1c8b1371ef6aa4187dae4495bd233c35/oneflow/core/cuda/softmax.cuh)，commit `25c8978c1c8b1371ef6aa4187dae4495bd233c35`。
- 官网检索内容与源码均支持三类路径；知乎全文访问受限。旧文章阈值和固定源码的分派不能当作同一版本配置，尤其不能直接用于现代 TileLang kernel。
- 源码定位：`SoftmaxWarpImpl`、`SoftmaxBlockSMemImpl`、`SoftmaxBlockUncachedImpl` 及 dispatch。

### S17

**BBuf 的可检查比较与逐步优化记录。**

- [Softmax 分析及测试表](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/15ea014a0ec19d9df9120dac87ad195862e18a29/cuda-kernels/softmax/README.md)；[完整目录](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/15ea014a0ec19d9df9120dac87ad195862e18a29/cuda-kernels/softmax)。
- [Reduce 逐步优化记录](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/15ea014a0ec19d9df9120dac87ad195862e18a29/cuda-kernels/reduce/README.md)；commit `15ea014a0ec19d9df9120dac87ad195862e18a29`。
- Softmax 比较为 A100 PCIe 40GB、FP32，选定 batch=32、heads=64、seq_len=16/32/64/128/512。FasterTransformer 基线取自 release/v1.0_tag 并移除 mask；只支持这组简化语义的比较，不评价当前框架整体。
- Reduce 记录提供多版本与 profiler 对照，但不将其图中各项带宽数字作为本库定量结论，也不默认旧版 warp 同步写法能跨代复用。
- 知乎关联入口：[OneFlow 原文](https://zhuanlan.zhihu.com/p/341059988)、[Reduce 系列](https://zhuanlan.zhihu.com/p/426978026)。正文分析主要依据可读取的仓库内容。

### S18

**NVIDIA CUTLASS：流水线的收益与存储成本。**

- [固定 Efficient GEMM 文档](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/media/docs/cpp/efficient_gemm.md)，commit `e13f5cd4142c43efc37c2f6339923eb4a57afb99`；定位 Pipelining、Parallelized Reductions。
- [NVIDIA 作者文章](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)说明 shared 与寄存器双缓冲及额外容量成本。
- 使用的是机制说明，不将该文旧硬件数据当成 TileLang 或当前 GPU 的性能预测。

### S19

**NVIDIA 转置示例：global 合并访问与 shared bank conflict 分开验证。**

- [作者原文](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)，2013-02-18；本轮读取日期 2026-10-29。
- 依次比较 naive、shared 重排、padding；测试硬件为 Tesla M2050/K20c。只用其机制和验证思路，不搬用旧速度结论。
- 知乎补充讨论见 Z02；实际 shared 冲突仍需按目标指令的事务分组判断。

## 待核验的知乎线索

以下条目用于指导后续调查，不参与正式性能结论，也不把搜索片段当全文。

### Z01

[暮易：Flash Attention CUDA Kernel 优化 G2——从算术指令到 Occupancy](https://zhuanlan.zhihu.com/p/2066951545090991335)。

知乎官方 CLI 返回片段包含逐版本对比、算术优化无收益和双缓冲资源退化。未获得完整原文、固定代码与原始 profiler 产物；只保留“检查收益是否被资源成本抵消”的待验证线索，不引用其具体吞吐/stall 数字。

### Z02

[知乎：CUDA 的 swizzle 是怎么实现 bank conflict free 的？](https://www.zhihu.com/question/667972067)。

网页检索内容包含 padding、swizzle、指令布局的讨论，完整回答访问受限。属于多人回答页，不将页面内各观点合成一个作者结论；基础机制用 S19 及目标 GPU 官方指令文档核对。

## 检索与收录原则

知乎 CLI 返回过一组结果，其他请求触发限流后停止，改用网页搜索和作者源码追溯。科普、转载、框架宣传和缺失测试条件的“几倍加速”不作为正式证据；非 GPU 后端不纳入。FlashQLA 按本库既有范围不收录。

本次选中的资料只支持七篇模式的决策结构，不能据此断言已覆盖全部 GPU 优化类型。新增实测应记录目标 GPU、软件版本、完整语义/shape、计时范围与原始产物，并回填反例。

## 本轮检查记录

- 15 条正式 PR 的合并状态和 merge SHA 与 GitHub API 一致。
- 24 个登记/核查的固定源码路径已通过 GitHub API 验证存在；包括 PR 代表文件和 OneFlow、BBuf、CUTLASS 资料。
- 7 篇模式的必要段落、唯一编号，以及涉及的 12 个 Markdown 文件内 98 个本地链接/锚点检查通过。
- 未运行 TileLang 编译、GPU 正确性或性能测试；本轮没有把上游测量升级为本地验证。

[返回模式索引](README.md)
