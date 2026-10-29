# 来源与收录边界

核查日期：2026-10-29。网络检索用于发现候选，代码事实以以下固定 commit 的一手源码为准。**33 篇是本轮精选覆盖，不宣称穷尽全网或全部上游算子，也不是性能排行榜。**

## 固定版本

| 来源 | commit | 案例数 | 许可证 |
| --- | --- | ---: | --- |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2a6c2045705bd2ac46136cece18f0d32c9c13650) | `2a6c2045705bd2ac46136cece18f0d32c9c13650` | 10 | [原文](_licenses/sglang.txt) |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/fa3b9d1377219247060d3c8b49eae199c9e7e154) | `fa3b9d1377219247060d3c8b49eae199c9e7e154` | 3 | [原文](_licenses/vllm.txt) |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/0a8c6716513bcdf8a728735c531c210c33077784) | `0a8c6716513bcdf8a728735c531c210c33077784` | 11 | [原文](_licenses/tilelang.txt) |
| [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels/tree/c1cf914a82e6901c679445f872051ce44ac2cc63) | `c1cf914a82e6901c679445f872051ce44ac2cc63` | 6 | [原文](_licenses/tilekernels.txt) |
| [tile-ai/TileOPs](https://github.com/tile-ai/TileOPs/tree/2c54caece3bc298b8794a364f8ed0c126fd7289d) | `2c54caece3bc298b8794a364f8ed0c126fd7289d` | 3 | [原文](_licenses/tileops.txt) |

五个仓库的固定版本文件树均已获取并检查为未截断。文件树检索用于确认路径，**不等于逐个阅读或评测了仓库中全部文件**。本轮逐篇阅读与整理的源码、符号及测试/参考入口在各案例末尾登记。

## 为什么收录这些实现

- **SGLang（10）**：BF16→FP8 激活量化，MiniCPM pooling/Top-k，DSA 的基础/手动流水/分段 decode/分页打分，QSA 打分，DSV4 FP8 cache，以及两类 mHC 融合。区分 attention 输出与索引器输出。
- **vLLM（3）**：mHC pre+RMSNorm、带适用性选择的 post/pre 融合、head 融合。与 SGLang 对照线程分工、分派条件与数值语义。
- **TileLang 官方（11）**：普通/persistent/intrinsics GEMM、低比特 GEMM/GEMV、MHA 前后向、变长 GQA、MLA decode、RMSNorm 与 online softmax。用于可读的机制基线，不能因为“官方”就推断最快。
- **TileKernels（6）**：per-token/per-block cast、SwiGLU+quant、Engram gate、Sinkhorn 前后向与小 k Top-k。保留模型特定布局和舍入限制。
- **TileOPs（3）**：带选择/拒绝条件的 RMSNorm，常驻与流式 softmax，add+RMSNorm+quant。重点提取“何时选择”的实现依据。

收录标准：实际包含 TileLang 核心；至少有一个独立的设计取舍；能界定计算语义与输入条件；代码与来源可追溯。已有相关测试入口是线索，运行结果才是证据。

## 去重与未独立收录

| 项目/变体 | 本轮处理 | 理由 |
| --- | --- | --- |
| FlashQLA | 不收录 | 按本次约定排除 |
| 同一源码的不同 shape、stage、dtype 配置 | 并入相同设计案例 | 配置数量不等于独立经验数量 |
| SGLang/vLLM 同义 mHC post 基础实现 | 不各建一篇 | 先覆盖更有决策价值的 pre+norm、跨阶段融合与 head；基础 post 可从固定来源比较 |
| DSA FP8 partial 与 BF16 partial 的同家族变体 | 不另建 BF16 案例的复制版 | 本轮由 DSV4 专用 FP8 cache 案例覆盖格式相关设计；两种 cache 契约不能互换 |
| 官方 INT4 Ladder/LOP3 高阶变体 | 仅在来源中保留入口 | 附带离线权重变换与 BitBLAS 依赖，需另立完整格式契约；基础解包 GEMM 和可选 LOP3 GEMV 已收录 |
| TileOPs softmax 的其他 split/cluster 变体 | 不混入常驻/流式案例 | 要独立说明跨 CTA 协议与调度，当前条目聚焦两种行存储策略 |
| Triton、CUDA、DeepGEMM 等非 TileLang 核心 | 可作后续对照基线，不冒充 TileLang 案例 | 以核心实现语言判定，不能只因代码出现在同一仓库而收录 |
| 未合并 PR、二手帖子中的速度说法 | 不当成当前实现与验证结论 | 分支状态、环境或测量口径不一定与固定源码一致 |

## 依赖与可复现边界

每篇代码是选定完整函数/计算链及模块内依赖的摘录，**不是把上游工程重打包成独立 Python 脚本**。函数体不省略；框架公共 helper、C++ header、编译器与 runtime 仍使用对应上游环境。

关键依赖入口：

- [低比特解包与 LOP3 helper](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/dequantize_gemm/quantize/__init__.py)。
- [量化配置、scale、packing 公共实现](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/quant/common.py)。
- [缓存策略外部辅助](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/csrc/streaming_load.h)。
- [跨 CTA partial 通信](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/csrc/cluster_partials.h)。
- [vLLM TileLang JIT 接口](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/tilelang_utils/__init__.py)。

代码摘录保留原有文件头版权/SPDX 声明，各上游许可证原文存于 `_licenses/`。本库的解释文字不改变上游代码许可；TileLang LICENSE 中的附加历史说明也原样保留。

## 本轮检查记录

- 每篇代码块通过 Python 语法检查；所选 AST 节点与固定来源对应节点一致，未执行上游代码。
- 本地 Markdown 链接、索引覆盖、固定来源路径与许可证归属经过检查。
- 未进行 TileLang JIT 编译、GPU 正确性、性能或 profiler 检查。源码中的测试函数、原注释中的调优结论和可运行的 benchmark 入口，都不能转换成“本地测试通过”。
- 当前“下一步验证”是待执行的建议；没有把猜测填写成已定位的性能瓶颈。

[返回案例总索引](README.md)
