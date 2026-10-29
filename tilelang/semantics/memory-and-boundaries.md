# 内存作用域、初始化与尾块

> 关键词：alloc_shared / alloc_fragment / alloc_var / clear / fill / mask。
> 适用：GPU；不同操作的边界处理必须逐项检查；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先确定谁拥有数据、谁能读到数据，以及每个被读取的位置是否已经得到正确初值。**

## 语义与约束

| 存储 | 逻辑含义 | 必须检查 |
| --- | --- | --- |
| Tensor 参数 | 调用者提供的设备数据及其 shape/dtype/stride | 实际输入是否满足声明；输出是否全部写入 |
| `T.alloc_shared` | block 内可共享的临时存储 | 写读顺序、同步、复用时机 |
| `T.alloc_fragment` | 逻辑 tile 按布局分配到线程的局部元素 | 不是每个线程各有一份完整 tile；检查布局与初始化 |
| `T.alloc_var` | DSL 的可变标量存储 | 初值与跨迭代更新依赖 |

分配不等于清零。累加前使用适合语义的初始化；`T.clear` 是置零，`T.fill` 可表达其他值。

安全访存和正确的数学 padding 是两件事。标准编译流程可为部分访问补充边界保护，但不能将这推广到任意间接索引、外部指针、显式 intrinsic 或关闭相关 pass 的程序。

## 最小检查

运行 [向量加](../api/kernel-and-buffers.md)，将 N 依次改为 1、255、256、257、1031。该示例显式使用 `if gi < N`，不依赖读者猜测自动 guard 的范围。

对于归约，先写出无效元素的语义：sum 为 0，max 为负无穷，min 为正无穷；softmax 的无效 logit 通常需要负无穷。**全 mask 行**还需另行定义输出，不能认为负无穷 padding 已覆盖这一语义。

## 为什么这样写

错误的填充值也可能完全不越界，却悄悄改变归约结果。初始化和 mask 应跟随数学算子确定，不能仅以编译通过为依据。

## 常见误用与检查

| 症状 | 待检查原因 | 检查动作 |
| --- | --- | --- |
| 仅尾块错误 | 漏写输出或填充值错误 | 对比整 tile / tile±1，逐项检查最后一块 |
| 相同输入重复运行不同 | 未初始化或并发错误 | 显式初始化，再检查同步；不能只归因于浮点误差 |
| 全负输入的 max/softmax 异常 | 用 0 填充无效元素 | 构造全负行并对照数学参考 |
| 转置/切片输入错误 | stride 声明与实际数据不一致 | 先验证连续输入，再单独扩展非连续契约 |

## 来源与关联

- [docs/programming_guides/language_basics.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/language_basics.md)
- [tilelang/language/allocate.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/allocate.py)
- [testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py)
- [testing/python/transform/test_tilelang_transform_verify_buffer_init.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_verify_buffer_init.py)
- [数值排查](../debugging/wrong-results.md) · [online softmax 案例](../../kernels/reduction/tilelang-online-softmax.md)
