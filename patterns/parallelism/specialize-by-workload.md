# P04 按负载分派与选择缓存

**当不同形状的最优策略不同，用少量可验证分支覆盖差异，并把编译与预热计入成本。**

## 触发证据

- 随行宽、token 数、对齐或 dtype 变化，性能存在稳定交叉点或断崖。
- 整行常驻 fragment 时寄存器/spill 增多，或 shared 容量限制可启动/驻留能力。
- prefill/decode、小/大 batch、不同 GPU 的胜出策略不同。

先排除计时噪声、首次 JIT、缓存状态和未调优基线；这些差异不能直接拿来拟合阈值。

## 决策与 TileLang 落点

| 情况 | 候选分支 | 需要一起计算的成本 |
| --- | --- | --- |
| 短行，寄存器容量可接受 | 整行常驻，warp 内归约 | 行数是否足够并行 |
| 中等行，shared 容量可接受 | block 协作并缓存中间值 | barrier、shared 与驻留限制 |
| 宽行常驻成本过高 | 分段累积，再读取或重算 | 额外访存/算术与真实缓存命中 |
| 同一实现无法覆盖不同负载 | 少量 tile/线程/算法分支 | JIT 数量、warmup、graph、分派维护 |

TileLang：分别构造常驻和分段实现，在 host wrapper 中使用已知 shape/设备属性分派；不要为获取分派条件引入不必要的设备同步。每个编译变体都要覆盖实际 graph capture 前预热。

完整代码：[官方 RMSNorm 常驻/重读](../../kernels/normalization/tilelang-rmsnorm-resident-vs-chunked.md)、[TileOPs softmax](../../kernels/reduction/tileops-softmax-resident-streaming.md)、[RMSNorm 分派](../../kernels/normalization/tileops-rmsnorm-shape-dispatch.md)。

## 证据与反例

- **官方解释与源码 / CUDA**：[S16](../SOURCES.md#s16) OneFlow softmax 包含 warp 常驻、block shared 缓存和 block 重读策略。旧文章的 1024/4096 阈值依赖其实现与环境，不是 TileLang/现代 GPU 的通用参数。
- **上游比较 / CUDA**：[S17](../SOURCES.md#s17) BBuf 在 A100 PCIe 40GB、FP32 下比较 OneFlow 与早期 FasterTransformer 提取实现；后者的 mask 被移除。可用于研究相同简化语义，不能宣称整个框架普遍更快。
- **源码事实 / CUDA**：[S13](../SOURCES.md#s13) vLLM 在 H100 上按 token 数选择每 warp 的 head 工作量；单 head/warp 不适合所有 batch。
- **交叉点 / TileLang**：[S10](../SOURCES.md#s10) GB300、hc=4、hidden=5120，融合相对调优未融合链在 32 tokens 接近持平，48 tokens 已变慢；选择阈值必须比较合理基线。同时需要预热真实 split 参数。

## 最小验证与退出

1. 先比较 2–3 个有机制差异的候选；扫描边界前后及生产常见形状，不只测幂次方。
2. 分开报告正常执行、CUDA Graph、冷/热缓存；重新检查真实生成代码的寄存器/shared 使用。
3. 记录每增加一个分支的编译数、启动时间、预热与 graph 内存成本；保留通用回退。

**退出**：交叉点落在噪声范围；阈值对轻微输入/环境变化不稳定；分支收益不足以抵消维护和启动成本。迁移 GPU 或升级编译器后重新测量，不继承旧阈值。

[模式索引](../README.md)
