# 从 TileLang 到生成代码：在哪一层检查

> 关键词：compile_only / get_kernel_source / TL_LOWER_TRACE / IR / codegen。
> 适用：CUDA 示例；compile-only 需要包含 CUDA codegen FFI 的安装包；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**把“生成了源码”“设备编译通过”“设备运行正确”分别记录。**

## 最小完整示例

保存为 `compile_add.py`。模块顶层只定义 kernel；compile-only 会导入此文件，所以不要在顶层分配或运行 GPU Tensor。

```python
import tilelang
import tilelang.language as T

@T.prim_func
def add(A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32")):
    with T.Kernel(1, threads=64):
        for i in T.Parallel(64):
            C[i] = A[i] + B[i]
```

```bash
python -I -m tilelang.tools.compile_only --target '{"kind":"cuda","arch":"sm_80"}' --output_file add.cu compile_add.py
```

预期退出码 0 且生成 `add.cu`。该工具在 `enable_device_compile=False` 处停止；不运行 GPU，也不证明 NVCC 或 GPU 正确性。macOS 的某些 wheel 不含 CUDA codegen FFI；因此“无需 GPU”不等于任意安装都能生成 CUDA。

## 如何检查

| 层次 | 观察对象 | 能回答什么 |
| --- | --- | --- |
| 原始 TileLang / IR | 索引、shape、scope、依赖 | 数学语义和约束是否按预期表达 |
| lowering 前后 IR | 布局、guard、同步、流水变化 | 哪一步引入/丢失了目标结构 |
| 生成设备源码 | 地址计算、指令封装、同步 | 预期路径是否真的生成 |
| 设备编译与执行 | 编译日志、产物、实际输出 | 工具链能否接受，目标 GPU 上是否正确 |

已有编译后的 factory kernel 可用 `kernel.get_kernel_source()` 查看源码。若需追踪哪个 pass 改了代码，对 [vector_add.py](../api/kernel-and-buffers.md) 使用：

```bash
TILELANG_DISABLE_CACHE=1 TL_LOWER_TRACE=both python vector_add.py
```

在新进程中触发编译，按日志找 trace 产物；缓存命中可能没有完整 lowering。这个快照推荐 IR Lower Trace，旧 Pass Diff 仅保留兼容用途。调试输出关闭后再测性能。

## 常见误用与检查

- 只搜到某条指令名称就声称提速：还需确认执行路径、资源成本和时间。
- 源码有差异就认为语义改变：对照数据流与输出，文本差异本身不是证明。
- 对最终报错 pass 直接归责：沿 trace 找首次出现不合理 IR 的位置，错误可能由上游生成。

## 来源与关联

- [docs/tools/compile_only.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/compile_only.md)
- [docs/tools/lower_trace.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/lower_trace.md)
- [docs/tools/pass_diff.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/pass_diff.md)
- [故障分诊](../debugging/failure-triage.md) · [特化缓存](specialization-and-cache.md)
