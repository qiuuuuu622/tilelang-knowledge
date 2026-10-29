# tilelang-knowledge

面向人和 Agent 的 TileLang 知识库：理解实现为何这样设计、何时适用，并通过实验沉淀可复用经验。

已整理来自 SGLang、vLLM、TileLang、TileKernels、TileOPs 的 **33 篇单文件算子案例**，包含完整选定核心代码、设计取舍、适用边界和固定来源。见 [案例索引](kernels/README.md)。另有 **7 篇 GPU 优化决策模式**，结合已合并 PR、作者源码和社区复盘，说明触发证据、最小实验与退出条件。见 [模式索引](patterns/README.md)。当前完成资料与源码静态核查，尚无本地 GPU 实验结果；TileLang 编程知识目录仍待补充。

## 内容地图

| 目录 | 内容 |
| --- | --- |
| [tilelang/](tilelang/) | 语义、API、布局、编译、调试与调优 |
| [kernels/](kernels/) | 高性能算子实现与设计解释 |
| [patterns/](patterns/README.md) | 7 篇有适用条件、证据与退出条件的 GPU 优化模式 |
| [workbench/](workbench/) | 具体任务的证据、瓶颈假设、实验与下一步建议 |

### TileLang 编程知识

- [semantics](tilelang/semantics/)：执行语义、数据作用域、同步与依赖。
- [api](tilelang/api/)：常用 API、约束与最小示例。
- [layouts](tilelang/layouts/)：布局表达、推导与检查。
- [compilation](tilelang/compilation/)：编译流程、生成代码与编译问题。
- [debugging](tilelang/debugging/)：正确性问题与排查方法。
- [tuning](tilelang/tuning/)：性能测量、分析与调参方法。

### 案例与优化模式

算子案例按 GEMM、attention、reduction、normalization、quantization 分类。
优化模式按数据搬运、并行策略、融合、热循环、流水线五类管理。先按 [触发证据](patterns/README.md#按当前证据选择) 选择模式，再读完整案例；固定来源与未验证范围见 [模式来源登记](patterns/SOURCES.md)。当前模式证据覆盖 NVIDIA CUDA 和 AMD ROCm，不纳入 CPU、NPU、TPU；非 TileLang 实现的经验仅作为迁移建议。

## 使用方法

1. 阅读 [workflow.md](workflow.md)，沿工作流查找知识、案例与优化模式。
2. 将 `workbench/_template/` 复制为 `workbench/<task-id>/`，填写任务与当前状态。
3. 记录每轮决策、实现和运行证据，将有依据的结论回写到知识、案例或模式中。

Agent 使用和更新本仓库时遵循 [AGENTS.md](AGENTS.md)。
空目录中的 `.gitkeep` 仅用于让 Git 保留目录，可在添加内容后移除。
