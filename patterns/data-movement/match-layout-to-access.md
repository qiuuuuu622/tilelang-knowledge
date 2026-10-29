# P02 匹配数据布局与访问

**让物理布局服务实际读写方式，同时计算布局转换、资源和所有消费者的总成本。**

## 触发证据

- producer 后紧跟 transpose、contiguous、scale 重排或 padding kernel。
- global 访问事务多于有效数据需要，或 shared 访问有可定位的 bank conflict。
- 计算阶段需要的 fragment/指令布局与当前存储布局不匹配。

bank conflict 必须按实际指令及其事务分组分析；不能只数整 warp 覆盖了哪些 bank，也不能将所有向量访问都判成冲突。

## 决策与 TileLang 落点

| 层次 | 候选改动 | 成本与边界 |
| --- | --- | --- |
| 算子之间 | producer 直接写消费者格式，用合法 stride view 接入 | producer 的写入可能变差；其他消费者可能仍需原格式 |
| 补齐与量化 | 同一 kernel 写有效数据及消费者要求的 padding | scale、打包数据、零点的补齐语义要分别确认 |
| 算子内部 | 调整 shared/fragment layout、padding 或 swizzle | 容量、对齐、地址计算和目标指令约束一起变化 |

TileLang：检查 `T.copy`、`T.gemm` 与布局推导是否一致；必要时显式标注布局，再检查生成代码。不要只改变 Python tensor 的 shape 而假设物理数据已转置。

完整代码：[显式 GEMM 指令与布局](../../kernels/gemm/tilelang-gemm-intrinsics.md)、[per-block cast](../../kernels/quantization/tilekernels-per-block-cast.md)。

## 证据与反例

- **上游测量 / ROCm FP8 调用链**：[S04](../SOURCES.md#s04) 在 gfx95 且 M≥2 时让生产者输出转置 scale 并接零拷贝视图；M=1 保留原路径。MI355X、GLM-5.3 MXFP4 TP8 EAGLE、指定 C6 测试中，每步少 78 次 launch，step 17.175→16.943 ms。不是通用 GPU 或 TileLang 性能结论。
- **源码事实 / CUDA**：[S05](../SOURCES.md#s05) NVFP4 quant 直接生成 padded 输出，删除独立 padding copy；特定 TRTLLM 8×4 scale 布局未支持该补齐路径，不能省略后端判断。
- **反例 / CUDA**：[S06](../SOURCES.md#s06) fused act+quant 原先没写完整 swizzled scale padding，下游读取旧内存造成 NaN。修复在 kernel 内初始化这些位置，仍保留 `empty` 分配。
- **机制依据**：[S19](../SOURCES.md#s19) NVIDIA 转置示例分别解决 global 合并访问与 shared bank conflict。其旧 GPU 数字不能作为现代 Tensor Core 布局的收益预估；知乎 swizzle 讨论只作补充线索。

## 最小验证与退出

1. 固定算法与调度，只改一种布局；覆盖非连续输入、非整 tile、M=1、padding 行列。
2. 检查 producer 和每个 consumer 的地址映射，验证全链路数值；未写区域做污染测试。
3. 测完整链及资源变化，确认转换被消除后没有在另一处重现。

**退出**：无法满足指令对齐/布局约束；其他消费者需要昂贵反转换；shared 占用或写入成本吞掉收益。不要根据 bank conflict 指标单独宣布成功。

[模式索引](../README.md)
