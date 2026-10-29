# T.gemm 的形状、累加器与调用约定

> 关键词：T.gemm / transpose_B / clear_accum / alloc_fragment。
> 适用：示例为 CUDA FP16 输入、FP32 累加，建议先在 SM80 兼容环境检查；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先满足矩阵语义、局部 tile 和累加器约束，再选择流水或底层指令。**

## 语义与约束

普通稠密 GEMM 的有效矩阵维应匹配 A[M,K]、B[K,N]、C[M,N]；转置标志改变逻辑解释，不会自动更换输入 Tensor 的实际存储。此快照检查局部 tile 的 M/N/K 为静态维度；全局尺寸与局部 tile 不要混淆。

下面每轮对 C 累加，进入 K 循环前清零一次。若每轮重置 C，只剩最后一个 K tile 的结果。`T.gemm` 具体选用哪种实现与目标、dtype、布局和配置有关。

## 最小完整示例

保存为 `small_gemm.py`。固定整 tile 形状，先隔离矩阵乘语义；非整 tile、低精度和手工流水均不在本示例验证范围。

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target="cuda")
def make_gemm():
    M, N, K = 128, 128, 64
    BM, BN, BK = 64, 64, 32
    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(N // BN, M // BM, threads=128) as (bx, by):
            a = T.alloc_shared((BM, BK), "float16")
            b = T.alloc_shared((BK, BN), "float16")
            c = T.alloc_fragment((BM, BN), "float32")
            T.clear(c)
            for k in T.serial(K // BK):
                T.copy(A[by * BM, k * BK], a)
                T.copy(B[k * BK, bx * BN], b)
                T.gemm(a, b, c)
            T.copy(c, C[by * BM, bx * BN])
    return gemm

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    c = torch.empty(128, 128, device="cuda", dtype=torch.float16)
    torch.backends.cuda.matmul.allow_tf32 = False
    make_gemm()(a, b, c)
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("PASS")
```

## 为什么这样写

serial 基线减少流水变量；FP32 累加与 FP16 输出分别明确。参考在 FP32 计算后转回输出 dtype；示例容差只用于这个输入范围，不应复制为所有算子的验收标准。

## 常见误用与检查

| 症状 | 检查动作 |
| --- | --- |
| K shape check failed | 展开转置后的逻辑尺寸，核对复制到 shared 的形状 |
| 结果像最后一块乘积 | 检查 clear 的位置与 `clear_accum` 使用 |
| shape 合法仍不能 lowering | 检查 dtype、scope、tile、线程与指令布局约束 |
| 改 FP8/INT4 只改 dtype | 量化比例、打包格式与支持指令需要独立设计 |

## 来源与关联

- [tilelang/language/gemm_op.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/gemm_op.py)
- [examples/gemm/example_gemm.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm.py)
- [流水 GEMM 案例](../../kernels/gemm/tilelang-gemm.md) · [布局推导](../layouts/inference-and-annotations.md)
