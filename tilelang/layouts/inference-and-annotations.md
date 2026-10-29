# 何时依赖推导，何时标注布局

> 关键词：annotate_layout / layout inference / loop_layout / GEMM / reduction。
> 适用：GPU；约束依赖具体操作与架构；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先保留自动推导的正确基线；只有明确希望控制的映射和消费约束后，才添加布局标注。**

## 语义与约束

布局推导需要协调同一数据的生产者和消费者：copy 的源与目标、逐元素循环、GEMM 与归约可能同时约束一个 buffer。`T.annotate_layout` 为特定 buffer 提供布局信息；`T.Parallel` / 普通 SIMT copy 的 `loop_layout` 则控制相应循环映射，两者不是任意互换的开关。

此快照的普通 copy `loop_layout` 仅适合 SIMT copy 路径，不能直接套到 TMA/LDSM/STSM/TMem 路径。明确约束应从 API 与生成代码检查，不从函数名猜测。

## 最小检查

以 [GEMM 完整示例](../api/gemm.md) 为基线，按以下顺序操作，避免一次修改多个约束：

1. 固定目标、shape、dtype、tile 和 threads，保存可编译版本的推导布局与生成源码。
2. 写出需要改变的映射：哪个 buffer，逻辑索引如何映射，目标是匹配谁的读取方式。
3. 只对这一个 buffer 添加 `T.annotate_layout`，保留其余推导；重新编译并执行正确性检查。
4. 如果失败，删除该标注确认恢复，再检查第一个冲突的操作；不要同时更改多个 pass 或屏蔽校验。
5. 若标注正确但产生额外 layout conversion，记录这些搬运；是否值得保留由端到端 kernel 时间决定。

## 为什么这样做

局部看起来连续的布局可能不满足矩阵指令的线程分布；减少一处冲突也可能增加另一处转换。需要沿数据流检查布局，而不是孤立选择一个“快的 swizzle”。

## 常见误用与检查

| 症状 | 待检查原因 | 下一步 |
| --- | --- | --- |
| 标注后 GEMM 无法 lowering | A/B/C 或线程分布不兼容 | 对照原推导布局，检查消费端要求 |
| fragment 在多个循环使用时报冲突 | 两个循环要求不同线程归属 | 找到冲突边，评估改映射或显式转换 |
| 换 GPU 后旧布局失效 | warp/wave 或指令规则不同 | 按目标重新推导，不能机械复制线程编号 |
| 正确但更慢 | 多余转换、非合并访问或资源增加 | 查看生成源码与 profiling，进入布局优化模式 |

## 来源与关联

- [tilelang/language/annotations.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/annotations.py)
- [tilelang/language/copy_op.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/copy_op.py)
- [tilelang/language/loop.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/loop.py)
- [testing/python/transform/test_tilelang_transform_layout_inference.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_layout_inference.py)
- [查看布局](inspect-layout.md) · [布局与访存模式](../../patterns/data-movement/match-layout-to-access.md)
