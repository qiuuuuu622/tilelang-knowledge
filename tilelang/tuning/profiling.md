# 从测量证据定位下一步检查

> 关键词：Analyzer / roofline / generated code / profiling / bottleneck。
> 适用：GPU；各指标须由目标对应工具提供，静态估计不等于实测；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先确认时间花在哪里，再用能区分假设的指标选择实验。一个低利用率数字不能单独命名瓶颈。**

## 证据层次

| 工具/材料 | 能得到什么 | 不能直接证明什么 |
| --- | --- | --- |
| 生成 IR/源码 | 访存、同步、指令与控制流结构 | 实际吞吐、缓存命中或延迟 |
| TileLang `Analyzer` | 可识别操作的静态工作量与 roofline 估计 | 完整硬件流量、实际瓶颈或加速比 |
| GPU profiler | 实际 kernel 时间及目标支持的资源/访存/执行指标 | 某个指标下降必然改善总时间 |
| 独立 benchmark | 明确输入与计时口径下的延迟 | 所有 shape、GPU 或服务场景都更快 |

此快照 Analyzer 的 FLOPs 主要识别 `T.gemm`，global bytes 主要识别跨参数 buffer 的 `T.copy`；直接 load/store、逐元素算术和其他 intrinsic 不全部计入。不能用它的 0 FLOPs 判断逐元素 kernel 没有计算。`CUDA("cuda")` 设备描述会查询可见 GPU；静态分析流程也可能需要运行时环境。

## 最小分析流程

1. 固定 [GEMM 示例](../api/gemm.md) 的输入与配置，先完成正确性和无插桩计时。
2. 计算理论工作量 2MNK，再检查是否与 Analyzer 对该 IR 的覆盖范围一致；把模型值标为估计。
3. 收集同一 kernel 的资源、访存、执行与等待证据；保留工具报告及采样方式。
4. 只提出一个可证伪的改动，复测总时间与相关指标；确认收益后再扩展 shape。

## 从证据到模式

| 当前证据组合 | 候选检查 | 关联模式 |
| --- | --- | --- |
| 存在不必要中间张量或无效区域计算 | 能否删掉工作或重读 | [减少无效工作](../../patterns/data-movement/eliminate-unnecessary-work.md) |
| 地址分析不连续且实际访存效率差 | 对齐、向量化、布局转换 | [匹配布局](../../patterns/data-movement/match-layout-to-access.md) |
| 工作量不足、CTA 数或资源限制可解释空闲 | tile、split、线程粒度 | [工作粒度](../../patterns/parallelism/choose-work-granularity.md) |
| 不同 shape 下最佳配置不同 | 形状分支与分派成本 | [工作负载特化](../../patterns/parallelism/specialize-by-workload.md) |
| 多个短 kernel 与中间搬运占主要时间 | 融合边界与资源成本 | [融合](../../patterns/fusion/fuse-with-explicit-boundaries.md) |
| 热循环存在重复索引或可移出工作 | 简化地址和不变量 | [简化热循环](../../patterns/compute/simplify-hot-loop.md) |
| 搬运等待与独立计算有重叠空间 | stage、同步和缓冲资源 | [流水预算](../../patterns/pipelining/budget-pipeline-resources.md) |

## 退出条件

指标改善但总延迟未改善，不记录为性能胜利；插桩改变行为时保留差异，回到正常执行复核。测量范围不包含的结论保持未知。

## 来源与关联

- [docs/tools/analyzer.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/analyzer.md)
- [docs/tools/lower_trace.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/lower_trace.md)
- [测量](measurement.md) · [模式索引](../../patterns/README.md)
