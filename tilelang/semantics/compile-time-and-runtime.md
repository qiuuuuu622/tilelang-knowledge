# Python 代码何时执行，设备代码何时执行

> 关键词：range / T.serial / T.const / T.dynamic / T.dyn / specialization。
> 适用：本快照的 DSL；示例采用 factory + PrimFunc 写法；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**区分 Python 构建程序、生成设备端控制流、执行设备程序这三个阶段。**

## 语义与约束

| 写法位置 | 行为 |
| --- | --- |
| 普通 Python factory 中计算 tile、dtype | 构建/特化时执行，结果进入生成程序 |
| kernel 内 `for i in range(n)` 或 `T.serial(n)` | 构造串行设备循环；不等于 Python 自动展开 |
| kernel 内遍历 list/tuple、`enumerate`、`zip` | 构建 IR 时展开，迭代对象必须有限且可在构建时取得 |
| comprehension 中的 `range` | Python 迭代，边界和过滤条件必须在构建时可求值 |
| kernel 内由设备数据决定的 `if` | 设备端条件，不应拿它控制必须在构建时执行的 Python 操作 |

Python list 不是设备数组；用运行时 TIR 索引访问 Python list 不会自动生成查表 kernel。跨迭代可变状态应依据 DSL 变量规则表达，不能把普通表达式别名当作可写设备存储。

## 最小检查

使用 [kernel 骨架](../api/kernel-and-buffers.md)：外层 `make_add(N, block)` 的整数用于特化形状和 tile，内部 `bx * block + i` 则形成设备索引。

- 构造 N=1031 和 N=2048 的 factory 结果，分别保存生成源码；不要声称它们共用一个动态 kernel。
- 若任务确需动态形状，再从本版本的 `T.dyn` / `T.dynamic` 使用方式引入符号，验证多组尺寸；tile 的指令约束仍需静态满足。
- 现有案例中的 eager 写法 `T.const` 与 `T.empty` 由 JIT 处理；不要直接与 factory 的 `out_idx` 返回约定混搭。

## 为什么这样设计

静态参数有利于专门化和检查，但形状组合多时可能增加编译和缓存成本。动态维度减少特化需求，却不意味着所有操作都支持动态局部 tile。是否动态化取决于工作负载，而不是代码看起来更通用。

## 常见误用与检查

| 误用 | 修正 |
| --- | --- |
| 把 Python 列表当设备查找表 | 用 Tensor/Buffer 或直接计算索引表达式 |
| 编译期展开中使用运行时 break 条件 | 将该循环改成设备串行循环，检查目标循环的语义 |
| 向特化参数传 list/dict | 改用语义稳定、可哈希的 tuple 等；不能只为通过 hash 而隐藏可变状态 |
| 用更换输入尺寸测试“动态”但没有看编译行为 | 同时记录特化参数、编译次数与生成代码 |

## 来源与关联

- [docs/programming_guides/python_compatibility.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/python_compatibility.md)
- [docs/programming_guides/language_basics.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/language_basics.md)
- [tilelang/jit/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/jit/__init__.py)
- [特化与缓存](../compilation/specialization-and-cache.md)
