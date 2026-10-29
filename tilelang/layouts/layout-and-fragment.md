# 区分存储布局与线程 fragment

> 关键词：T.Layout / T.Fragment / forward_thread_fn / forward_index_fn。
> 适用：布局对象为主机侧构造；应用到 GPU kernel 还需目标约束验证；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**Layout 回答逻辑元素放在哪里；Fragment 还回答由哪个线程持有，以及在线程内部放在哪里。**

## 语义与约束

| 对象 | 需要表达的映射 | 常见误解 |
| --- | --- | --- |
| `T.Layout` | 逻辑索引 → 存储索引 | 更换 layout 自动等价于执行了一次 transpose copy |
| `T.Fragment` | 逻辑索引 → 线程编号、线程局部索引，可含复制 | fragment 的逻辑形状就是每线程寄存器数组形状 |

可表示的映射不一定满足某个 GEMM、归约或搬运指令的消费约束。复制布局允许重复持有，但需要明确哪些线程写回，不能把所有重复都解释为错误，也不能让重复写发生冲突。

## 最小完整示例

安装本基线 TileLang 后可执行下面脚本；只构造和打印映射对象，不编译或执行 GPU kernel。

```python
import tilelang.language as T

transpose = T.Layout([4, 8], lambda i, j: (j, i))
fragment = T.Fragment(
    [4, 8],
    forward_thread_fn=lambda i, j: j,
    forward_index_fn=lambda i, j: i,
)
print(transpose)
print(fragment)
print("input shape:", list(transpose.get_input_shape()))
print("mapped shape:", list(transpose.get_output_shape()))
print("thread extent:", fragment.get_thread_size())
```

预期 transpose 将 (i,j) 映射到 (j,i)，输入形状为 4×8、映射形状为 8×4。fragment 的 8 个逻辑线程各持有 4 个元素；例如 (2,3) 对应 thread=3、local=2。**8 线程只是教学映射，不是推荐的 GEMM launch 配置。**

## 为什么先写小映射

小范围枚举可以检查覆盖、冲突和索引规律。复杂 swizzle 若无法先解释几个地址，很难判断是否满足生产者与消费者共同约束。

## 常见误用与检查

| 症状 | 检查动作 |
| --- | --- |
| 元素位置不对 | 区分逻辑转置、存储映射和实际 copy，逐项追踪一个元素 |
| layout 能构造但 kernel 编译失败 | 检查线程范围、指令与 buffer scope 的约束 |
| 图上出现覆盖/空洞 | 检查映射是否有意复制或稀疏，不能只凭图判定合法 |

## 来源与关联

- [tilelang/layout/layout.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/layout/layout.py)
- [tilelang/layout/fragment.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/layout/fragment.py)
- [推导与标注](inference-and-annotations.md) · [可视化检查](inspect-layout.md)
