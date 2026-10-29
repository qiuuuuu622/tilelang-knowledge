# TileLang 编程知识

按问题检索的 **20 篇知识卡**：语义与约束 → 最小例子或检查流程 → 为什么这样写 → 常见误用 → 固定来源。

本批沿用现有 TileLang 算子案例的源码快照，不等同于任意发布版；见[版本、来源与核验边界](SOURCES.md)。代码面向 CUDA，ROCm 的通用概念与迁移约束另行标明，未宣称跨 GPU 实测通过。不收录 CPU、NPU、TPU 算子内容。

## 按当前问题进入

| 当前问题 | 先读 | 下一步 |
| --- | --- | --- |
| 不知道如何写并调用 kernel | [kernel 与 Buffer](api/kernel-and-buffers.md) | [执行模型](semantics/execution-model.md) |
| 不清楚 Python 在哪一阶段执行 | [编译期与运行期](semantics/compile-time-and-runtime.md) | [特化与缓存](compilation/specialization-and-cache.md) |
| 尾块、归约或特殊输入不正确 | [内存与边界](semantics/memory-and-boundaries.md) | [结果排查](debugging/wrong-results.md) |
| copy/GEMM 不会用或报约束错误 | [复制与 view](api/copy-and-views.md) / [GEMM](api/gemm.md) | [布局推导](layouts/inference-and-annotations.md) |
| 不知道是否需要等待/同步 | [同步契约](semantics/synchronization.md) | [竞争与复现](debugging/races-and-reproducers.md) |
| 编译失败，不知道从哪里查 | [失败分诊](debugging/failure-triage.md) | [生成代码与 Lower Trace](compilation/inspect-generated-code.md) |
| 能运行，但不知布局是否合适 | [Layout 与 Fragment](layouts/layout-and-fragment.md) | [查看布局](layouts/inspect-layout.md) |
| 性能数字不稳定或缺少瓶颈证据 | [测量](tuning/measurement.md) | [Profiling](tuning/profiling.md) → [优化模式](../patterns/README.md) |
| 想搜索 tile/threads/stages | [Autotuning](tuning/autotuning.md) | 独立正确性回归与测量复核 |

## 语义

- [逻辑 tile、线程和循环](semantics/execution-model.md)：先确定工作分配，再理解线程映射。
- [编译期与运行期](semantics/compile-time-and-runtime.md)：Python 构建、设备控制流、特化与动态维度。
- [内存与边界](semantics/memory-and-boundaries.md)：scope、初始化、mask、尾块与填充值。
- [同步](semantics/synchronization.md)：普通复制、显式异步、可见性、缓冲复用与流水依赖。

## 常用 API

- [kernel 与 Buffer](api/kernel-and-buffers.md)：完整向量加、显式输出、调用与检查。
- [复制与 view](api/copy-and-views.md)：区域、extent、stride、共享内存复制示例。
- [GEMM](api/gemm.md)：shape、转置、累加器、完整整 tile 示例。
- [归约与数值](api/reductions-and-numerics.md)：dim、输出形状、初值、累加 dtype 和行归约示例。

## 布局

- [Layout 与 Fragment](layouts/layout-and-fragment.md)：存储映射与线程归属。
- [推导与标注](layouts/inference-and-annotations.md)：生产者和消费者的共同约束。
- [查看布局](layouts/inspect-layout.md)：直接绘图、推导可视化与工具限制。

## 编译

- [JIT 与目标](compilation/jit-and-targets.md)：版本、架构、后端、工具链与环境检查。
- [查看生成代码](compilation/inspect-generated-code.md)：compile-only、IR、源码与 Lower Trace。
- [特化与缓存](compilation/specialization-and-cache.md)：如何确认实际编译和测量的实现。

## 调试

- [失败分诊](debugging/failure-triage.md)：按最早失败阶段选择下一步检查。
- [结果错误](debugging/wrong-results.md)：错误分布、边界输入、初始化与数值判断。
- [竞争与复现](debugging/races-and-reproducers.md)：缓冲生命周期、稳定失败判据、AutoDD。

## 调优

- [测量](tuning/measurement.md)：计时边界、完整 Event 示例、波动与记录。
- [Profiling](tuning/profiling.md)：区分静态估计与实测，将证据连接到 7 个优化模式。
- [Autotuning](tuning/autotuning.md)：合法搜索、显式参考、固定输入、trial 与缓存边界。

## 怎样使用与补充

先读问题对应条目，再跳到完整算子或优化模式；不要求 Agent 每次加载整个目录。编程知识解释“如何正确表达和检查”，[算子案例](../kernels/README.md)解释完整实现，[模式](../patterns/README.md)解释何时尝试优化，具体实验进入 `workbench/`。

每篇保留：一句结论；适用版本/后端与关键词；语义和约束；最小完整示例或操作流程及预期；设计理由；常见误用；固定源码/测试来源与关联。按 API 家族组织，不逐个复制全部参数表。代码优先单页自包含；复用 `vector_add.py` 的页面必须明确指向保存步骤，不依赖未说明的外部文件。

新增复杂 TMA、warp specialization 或架构专用操作时，应由实际案例需求驱动，补齐同步契约和最小复现后再收录。未验证的性能判断链接到待验证模式，不写成语言保证。
