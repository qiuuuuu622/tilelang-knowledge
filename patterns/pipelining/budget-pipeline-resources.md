# P07 用资源预算选择流水线

**仅当可隐藏的等待超过额外缓冲、同步和资源成本时，增加流水深度或线程角色分工。**

## 触发证据

循环中装载与计算串行；后续 tile 的搬运可与当前 tile 计算独立执行；profiling 和依赖分析共同显示等待阻塞有效工作。低利用率或某个 stall 百分比不能单独证明流水线有效。

## 决策与 TileLang 落点

| 方案 | 适用前提 | 新增成本 |
| --- | --- | --- |
| 少量 stage 的预取/双缓冲 | 迭代有足够计算覆盖下一轮搬运 | shared/寄存器缓冲，prologue/epilogue |
| warp/线程组分工 | 搬运与计算可由不同角色推进 | barrier、角色资源分配、空闲线程 |
| 减少 stage 或缩小 tile | 资源限制驻留或出现 spill | 潜在的等待增加，需要重新测量 |

TileLang：先比较普通循环与 `T.Pipelined`，再考虑显式生产/消费角色。先固定 tile 与线程数，避免将多个因素混成一个结论。barrier 的 arrive/wait、phase、缓冲重用和所有尾部路径必须一起设计。

完整代码：[普通 GEMM](../../kernels/gemm/tilelang-gemm.md)、[DSA 基础版](../../kernels/attention/sglang-dsa-sparse-baseline.md)、[DSA 手动流水版](../../kernels/attention/sglang-dsa-warp-specialized.md)。

## 证据与反例

- **机制与实现 / CUDA**：[S18](../SOURCES.md#s18) CUTLASS 用 shared tile 和寄存器 fragment 的双缓冲重叠搬运与计算。额外存储是收益的成本，不是免费优化；本库关联的 DSA 案例也有 ready/free barrier 与双缓冲，但尚无本地性能测量。
- **正确性反例 / TileLang**：[S14](../SOURCES.md#s14) mHC fused RMSNorm 的特定循环在 stage=3、loop-carried sumsq 与持续使用的 shared buffer 组合下出现错误；改为 stage=2 修复。GB200 上作者报告不同 hidden size 和 token 数结果不同，不能只测 H=4096，也不能归纳成所有 stage=3 都错误。
- **跨 GPU 约束 / TileLang**：[S15](../SOURCES.md#s15) mHC 线程角色分组在 AMD wave64 下触发归约分组问题；改成完整 wavefront 对齐后修复，CUDA 路径不变。这是角色分工的迁移约束，不是 stage 深度的性能证据。
- **待核验反例**：[Z01](../SOURCES.md#z01) 知乎复盘描述双缓冲减少等待但寄存器成本使整体变慢；具体结果未获得原代码核验，不复制其速度数字。

## 最小验证与退出

1. 固定 tile/线程，比较串行和少量 stage；逐项记录每 CTA shared、每线程寄存器、spill、驻留能力与总延迟。
2. 覆盖循环次数 0/1/2、奇偶、多轮、尾部、全 mask、不同 batch；审查是否有不参与同步的分支。
3. 重复正确性检查与 graph replay；将 profiling 指标与正常执行时间分开，变化后重新定位瓶颈。

**退出**：等待下降但完整延迟上升；资源超限/spill；任何同步、数值或非确定性回归。先减 stage 或调整 tile，再考虑复杂的手动流水；不能通过放宽精度掩盖同步错误。

[模式索引](../README.md)
