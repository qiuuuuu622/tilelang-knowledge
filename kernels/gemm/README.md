# GEMM / GEMV

先读普通 GEMM，再根据瓶颈选择：调度/缓存问题看 persistent，指令或布局问题看 intrinsics，权重带宽问题看低比特解包。大 M 的 GEMM 与小 M 的 GEMV 分开评测。

INT4 示例主要展示解包机制，不自动涵盖真实模型的 group scale、zero-point 和权重变换。GEMV 案例中特别保留了上游检查过于宽松这一限制。

## 案例索引

共 5 篇。代码行数用于估计阅读成本；先读“什么时候选择”和“为什么这样设计”，匹配后再读代码。

| 案例 | 值得借鉴的设计 | 核心代码行数 |
| --- | --- | ---: |
| [TileLang 官方：分块与流水 GEMM](tilelang-gemm.md) | 用共享内存分块、FP32 累积和 K 维流水表达 FP16 矩阵乘，是后续优化的可读基线。 | 26 |
| [TileLang 官方：persistent GEMM 对照](tilelang-persistent-gemm.md) | 对照每 tile 一个 CTA 与按 SM 数启动 CTA、循环领取 tile 两种调度方式。 | 82 |
| [TileLang 官方：显式 Tensor Core 指令与布局](tilelang-gemm-intrinsics.md) | 显式组织片段布局、装载与 MMA 指令，展示高层 T.gemm 以下的控制点。 | 154 |
| [TileLang 官方：INT4 解包与 GEMM 融合](tilelang-int4-dequant-gemm.md) | 权重以低比特打包传入，在片上解包后直接参加 GEMM，避免完整反量化权重落到显存。 | 81 |
| [TileLang 官方：面向小 M 的 INT4 GEMV](tilelang-int4-dequant-gemv.md) | 把输出列分工、K 维线程归约和低比特解包融合，展示小 M 时不用常规大 GEMM tile 的选择。 | 157 |

## 证据边界

全部条目已静态核查固定版本源码、保留完整选定函数及模块内依赖；未运行 GPU 编译、正确性或性能测试。上游测试入口不代表本仓库验证通过。复杂模型案例依赖对应上游工程，不能直接作为独立脚本运行。

[返回总索引](../README.md) · [来源、版本与收录边界](../SOURCES.md)
