# 失败后先定位阶段，再修改代码

> 关键词：import / parse / layout inference / codegen / device compile / runtime。
> 适用：GPU；先保留完整失败输入、环境与日志；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先判断失败发生在哪个阶段，选择能缩小原因范围的检查。不要一次更改算法、布局和编译选项。**

## 分诊表

| 最早失败阶段 | 首先保存 | 优先检查 | 确认有效的信号 |
| --- | --- | --- | --- |
| import / 环境 | 异常、包路径、版本 | FFI、依赖、Python 环境 | 最小 import 与环境脚本通过 |
| Python/DSL 构建 | 完整函数、静态参数 | 不支持的语法、动态值进入 Python 操作 | 能形成预期 IR |
| shape / layout 检查 | 输入 shape/stride、目标、冲突操作 | tile 维度、转置、线程与指令约束 | 不屏蔽校验即可通过 |
| lowering / codegen | IR trace、pass 配置 | 首次异常转换、非法 scope/intrinsic | 生成结构符合预期的设备源码 |
| 设备编译 | 生成源码、编译命令、stderr | 工具链、架构、不支持的指令、资源限制 | 设备编译成功 |
| launch / 运行 | 输入、设备、错误出现位置 | 架构匹配、非法访问、同步 | 无运行错误且输出通过检查 |
| 结果不符 | 参考、误差位置、seed | 初始化、索引、mask、归约、数值 | 边界和压力输入均通过 |

## 最小操作流程

1. 用 [环境脚本](../compilation/jit-and-targets.md) 确认实际加载包与 GPU；用向量加确认基本链路。
2. 缩小到一个 shape、一组输入和一个 kernel，保留原始失败版本。
3. 编译阶段用 [compile-only / Lower Trace](../compilation/inspect-generated-code.md)；运行阶段进入 [结果排查](wrong-results.md) 或 [竞争排查](races-and-reproducers.md)。
4. 每次只修改能区分一个假设的因素；保存改动和结果。修复必须回归原始失败用例。

## 为什么这样排查

设备异步执行可能使异常在后续同步处才被观察到；最后一行报错不一定是产生问题的位置。编译器某个 pass 抛错，也可能是在拒绝更早构造出的非法程序。

## 停止误导性尝试

- 关闭边界、初始化或数据竞争检查后能编译，不算修复；必须解释原检查为何不适用并证明正确性。
- 放宽容差不用于解决越界、未初始化和同步错误。
- 缩小规模后不再失败，只说明触发条件变化；先保留差异，不宣布问题已解决。

## 来源与关联

- [docs/tutorials/debug_tools_for_tilelang.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tutorials/debug_tools_for_tilelang.md)
- [docs/tools/lower_trace.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/lower_trace.md)
- [testing/python/transform/test_tilelang_transform_verify_buffer_init.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_verify_buffer_init.py)
- [工作流](../../workflow.md) · [竞争与复现](races-and-reproducers.md)
