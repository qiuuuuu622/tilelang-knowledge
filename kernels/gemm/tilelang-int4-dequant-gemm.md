# TileLang 官方：INT4 解包与 GEMM 融合

权重以低比特打包传入，在片上解包后直接参加 GEMM，避免完整反量化权重落到显存。

## 什么时候选择

- **适用**：A[M,K]、按 N×K 打包的 unsigned 低比特 B；num_bits、packing 和 tile 整除关系符合代码。
- **不适用**：带 group scale/zero-point 的任意量化模型直接套用：此核心主要是整数解包，不是完整模型量化契约。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 反量化权重写回再读取成本高 | 局部解包后写共享内存，紧接 T.gemm | 省中间 HBM；解包与额外共享内存仍有代价 |
| 小整数逐个加载效率低 | 压缩权重按向量读取，再展开 | 降低权重读字节数；依赖 packing 对齐 |
| 解包与矩阵乘需要衔接 | K tile 流水组织读入、展开、GEMM | 可重叠部分工作；stage 数受资源限制 |

**下一步验证**：先逐元素核对 unsigned 解包和高低位顺序，再对照显式解包+GEMM；记录节省带宽与增加的解码指令。

## 核心完整代码

入口：`matmul`。完整基础 matmul。依赖同目录 quantize 包；更复杂的 Ladder/BitBLAS 变体在上游同文件，未混入此案例的计算契约。

```python
import tilelang.testing

from tvm import DataType

import tilelang.language as T

@tilelang.jit(out_idx=[2])
def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    num_bits=4,
):
    from quantize import _tir_packed_to_unsigned_convert

    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.int8
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    local_size_compressed = local_size // num_elems_per_byte

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_local([local_size_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([local_size], in_dtype)
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            tx = T.get_thread_binding()

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * local_size_compressed)):
                    for v in T.vectorized(0, local_size_compressed):
                        index = i * threads * local_size_compressed + tx * local_size_compressed + v
                        vi = index // (block_K // num_elems_per_byte)
                        vj = index % (block_K // num_elems_per_byte)
                        B_local[v] = B_shared[vi, vj]
                    for v in T.serial(0, local_size):
                        B_dequantize_local[v] = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                            num_bits,
                            B_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    for v in T.vectorized(0, local_size):
                        index = i * threads * local_size + tx * local_size + v
                        vi = index // block_K
                        vj = index % block_K
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

## 来源与验证

- 来源：[tile-ai/tilelang · example_dequant_gemm_fine_grained.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/dequantize_gemm/example_dequant_gemm_fine_grained.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L3–L3, L5–L5, L6–L6, L16–L90。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/dequantize_gemm/example_dequant_gemm_fine_grained.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
