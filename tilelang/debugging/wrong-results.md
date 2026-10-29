# 怎样区分索引错误、并发错误和数值误差

> 关键词：correctness / mask / initialization / tolerance / NaN / reference。
> 适用：GPU 输出检查；容差必须按算子契约制定；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先定位错误分布，再解释误差来源；随机输入下的一次 allclose 不是完整验收。**

## 最小完整检查

将 [向量加示例](../api/kernel-and-buffers.md) 保存为同目录的 `vector_add.py`，下面保存为 `check_add.py`。它只是向量加的回归入口，其他算子需更换参考和边界集合。

```python
import torch
from vector_add import make_add

torch.manual_seed(0)
for n in (1, 127, 128, 255, 256, 257, 1031):
    kernel = make_add(n)
    for mode in ("random", "negative", "zero"):
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)
        if mode == "negative":
            a, b = -a.abs(), -b.abs()
        elif mode == "zero":
            a.zero_()
            b.zero_()
        ref = a + b
        for repeat in range(3):
            out = torch.full_like(a, float("nan"))
            kernel(a, b, out)
            assert torch.isfinite(out).all().item(), (n, mode, repeat)
            torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
print("PASS: boundaries, data families, repeated runs")
```

预期输出 PASS；NaN 初值用于发现漏写，不作为所有算子的通用输入策略。原子累加输出若要求先清零，必须按自身契约初始化。

## 从现象到下一步

| 错误分布 | 待验证假设 | 下一步 |
| --- | --- | --- |
| 仅最后一行/列错误 | 尾部 mask、写回范围 | 测 tile±1，检查无效值与有效输出覆盖 |
| 固定位置错误 | stride、布局、转置 | 用递增索引输入追踪一个元素 |
| 相同输入每次不同 | 未初始化、竞争；也可能是允许的非确定性归约 | 先排初始化与同步，再依据数值契约解释 |
| 误差随归约长度增长 | 累加精度或顺序 | 用更高精度参考，逐段检查中间值 |
| NaN/Inf 特殊输入差异 | 特殊值传播、溢出、全 mask 定义 | 明确算子约定后逐项测试 |

## 验收边界

报告最大绝对误差、误差位置与相对误差时的近零处理；记录 atol/rtol 的依据。检查非连续输入前先确认 API 承诺支持，否则应明确拒绝而非默默接受。修改累加 dtype、近似函数或算法后，重新验证整个输入族。

## 来源与关联

- [testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py)
- [testing/python/transform/test_tilelang_transform_verify_buffer_init.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/transform/test_tilelang_transform_verify_buffer_init.py)
- [testing/python/language/test_tilelang_language_reduce.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/language/test_tilelang_language_reduce.py)
- [归约数值](../api/reductions-and-numerics.md) · [内存边界](../semantics/memory-and-boundaries.md)
