# 编程知识的版本、来源与核验边界

## 基线

- 上游：[tile-ai/tilelang](https://github.com/tile-ai/tilelang)。
- 固定 commit：[`0a8c6716513bcdf8a728735c531c210c33077784`](https://github.com/tile-ai/tilelang/tree/0a8c6716513bcdf8a728735c531c210c33077784)，与本库现有 TileLang 算子案例保持一致。
- 核查日期：2026-10-29。发布版标签、官网版本横幅或相同版本号均不能替代源码 commit。
- 范围：GPU 编程；首批可执行示例指定 CUDA。涉及 ROCm 的内容只覆盖明确的目标/迁移概念，不声称 CUDA 示例直接在 AMD 通过。
- 上游代码与文档的许可见[本库保存的 TileLang LICENSE](../kernels/_licenses/tilelang.txt)及[固定版本 LICENSE](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/LICENSE)。本批说明为重新组织的中文知识卡，代码为依据文档、源码和示例缩小或改写的教学例子，非上游测试原样执行报告。

## 一手资料如何使用

| 材料 | 用途 | 核心入口 |
| --- | --- | --- |
| 编程指南 | 概念、控制流、流水与公开契约 | [Language Basics](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/language_basics.md)、[Python Compatibility](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/python_compatibility.md)、[Instructions](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/instructions.md)、[Software Pipeline](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/software_pipeline.md) |
| API 与 JIT 源码 | 输入检查、默认行为、调用和输出约定 | [copy](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/copy_op.py)、[GEMM](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/gemm_op.py)、[reduce](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/reduce_op.py)、[JIT](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/jit/__init__.py) |
| 布局定义与测试 | 映射定义、推导约束 | [Layout](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/layout/layout.py)、[Fragment](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/layout/fragment.py)、[inference tests](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_layout_inference.py) |
| 编译与调试工具文档 | 操作步骤、产物和工具限制 | [compile-only](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/compile_only.md)、[Lower Trace](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/lower_trace.md)、[AutoDD](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/autodd.md) |
| 边界测试 | 已存在的合法/非法场景与回归思路 | [safe memory](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py)、[buffer init](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_verify_buffer_init.py)、[async copy SM80](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/language/test_tilelang_language_async_copy_gemm_sm80.py) |
| 调优源码与文档 | 参考比较、搜索、缓存与 trial 语义 | [autotuner](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/autotuner/tuner.py)、[profiler](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/profiler/__init__.py)、[Analyzer](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/analyzer.md) |

每篇末尾还有直接相关的固定路径来源。测试文件在上游存在，不表示本轮执行过该测试。社区教程用于发现问题和组织解释，API 契约最终回查上述一手来源；本批不把社区速度数字写成事实。

## 文档与源码交叉核查的具体取舍

| 容易误读的地方 | 本库采用的写法 |
| --- | --- |
| 入门文档对 copy extent 有较宽泛的描述 | 按 copy 源码明确：内部形状语法糖不等于一般广播；示例保持区域匹配 |
| Autotuning 指南提到参考可返回 bool | 默认 profiler 实际比较参考输出与 kernel 输出；示例返回 Tensor，并设置 out_idx，避免只返回 bool |
| `skip_check=False` 看起来像自动开启验证 | 源码仍要求存在 ref_prog；显式配置参考并检查实际 validation 状态 |
| 旧资料使用 Pass Diff | 本快照推荐 IR Lower Trace，保留旧工具只作版本兼容说明 |
| 布局图/roofline 结果看起来像性能结论 | 图只用于检查映射；静态模型不等于正确性或 GPU 性能测量 |

这些取舍针对固定快照；升级时重新核查，不声称上游所有历史版本行为一致。

## 本轮验证了什么

| 层次 | 状态 |
| --- | --- |
| 上游路径与关键契约核查 | 已完成；引用固定 commit，按文档/源码/测试交叉检查 |
| Markdown 内链、代码块与 Python 语法 | 20 篇知识卡及 2 篇索引/来源页：12 个 Python 代码块、126 个本地链接已检查；语法可解析不代表 TileLang DSL 支持 |
| TileLang lowering / 设备编译 | 未执行 |
| GPU 输出正确性 | 未执行；文中的 PASS 为预期，不是本地结果 |
| 性能与跨 GPU 兼容性 | 未执行；没有本地延迟或加速结论 |

后续在真实 GPU 上验证时，逐项补充设备、工具链、输入、命令、配置和产物链接；不要把某个环境通过扩大成所有架构保证。内容或核验状态改变时，更新相关条目而非只改本表。

[知识索引](README.md)
