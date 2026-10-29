# 逻辑 tile、线程和循环分别表达什么

> 关键词：T.Kernel / T.Parallel / T.serial / T.unroll / T.Pipelined。
> 适用：GPU 编程模型；线程数量与指令约束按具体后端检查；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先描述一个 block 负责的逻辑工作，再让布局和编译器决定元素如何分配给线程。**

## 语义与约束

| 表达 | 描述什么 | 不能直接推出什么 |
| --- | --- | --- |
| `T.Kernel(gx, gy, threads=128)` | GPU grid 与每个 block 的线程数 | 一个 block 只能处理 128 个元素 |
| `T.Parallel(M, N)` | M×N 个逻辑迭代，由布局分配执行 | 一个迭代固定等于一个硬件线程 |
| `T.serial(K)` | 有顺序的循环 | 所有计算一定只由一个线程完成 |
| `T.unroll(K)` | 请求展开循环 | 展开一定更快或不会增加代码体积 |
| `T.Pipelined(K, num_stages=s)` | 为迭代间搬运、计算调度提供流水信息 | s 越大越快，或任意依赖均可重排 |

## 最小检查

运行 [vector_add.py 完整示例](../api/kernel-and-buffers.md)：N=1031、tile=256、threads=128。应有 5 个 block，最后一个只有 7 个有效元素；不是 1031 个 block，也不是每线程恰好处理一个元素。

1. 保持 tile=256，只将 threads 从 128 改为 256，确认数学结果相同。
2. 查看生成源码中的线程索引与循环/向量访问；逻辑迭代可能由每线程多个元素实现。
3. 若换成 GEMM，重新检查该线程配置是否满足指令与布局约束；向量加可用的配置不是所有算子的合法配置。

## 为什么这样表达

tile 形状决定局部计算与数据复用；线程布局决定这些工作如何分配。两者分离，才允许同一个逻辑程序探索不同映射。先用结构化操作建立正确基线，再在有证据时接管线程映射。

## 常见误用与检查

| 症状 | 待检查原因 | 检查动作 |
| --- | --- | --- |
| 改线程数后布局推导失败 | 指令或 fragment 的线程约束不满足 | 还原线程数，检查消费者操作的约束 |
| 并行循环中递推结果错误 | 迭代存在数据依赖 | 将递推写为串行或显式归约，不让并行迭代互相读写 |
| 增加流水后只在多轮循环出错 | loop-carried 依赖或缓冲重用 | 退回 serial，覆盖短循环、奇偶和尾部 |

## 来源与关联

- [docs/programming_guides/language_basics.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/language_basics.md)
- [tilelang/language/loop.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/loop.py)
- [布局模型](../layouts/layout-and-fragment.md) · [工作粒度模式](../../patterns/parallelism/choose-work-granularity.md)
