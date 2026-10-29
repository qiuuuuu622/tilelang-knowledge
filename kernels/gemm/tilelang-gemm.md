# TileLang 官方：分块与流水 GEMM

用共享内存分块、FP32 累积和 K 维流水表达 FP16 矩阵乘，是后续优化的可读基线。

## 什么时候选择

- **适用**：A[M,K]、B[K,N] 的 FP16 GEMM，先从上游示例配置与受支持的 Tensor Core 硬件开始。
- **不适用**：已经要求量化、分组 GEMM、融合 epilogue 或特殊布局；未经测量就替换成熟 GEMM 库。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 每个输出重复读取输入 | A/B tile 放共享内存，T.gemm 复用 | 提高数据复用；tile 越大资源占用越高 |
| K 方向读写与计算串行 | T.Pipelined(num_stages=3) | 有机会重叠；stage 数增加共享内存 |
| FP16 长归约误差大 | C_local 使用 FP32 | 改善累积精度；最终仍转换为输出 dtype |

**下一步验证**：固定 M/N/K 扫 tile 与 stages，检查生成指令、寄存器和共享内存；先验证非整 tile 尾部，再比较同精度库基线。

## 核心完整代码

入口：`matmul`。完整 matmul；同文件 main 提供参考与评测入口，本文未运行。

```python
import tilelang

import tilelang.language as T

@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

## 来源与验证

- 来源：[tile-ai/tilelang · example_gemm.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L1–L1, L2–L2, L5–L26。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
