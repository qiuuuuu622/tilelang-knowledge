# 怎样查看自己写的布局与编译器推导的布局

> 关键词：plot_layout / TL_LAYOUT_VISUALIZATION_ENABLE / fragment visualization。
> 适用：直接绘图需 TileLang 与 Matplotlib；推导示例需 CUDA 编译环境；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**自己构造的布局图和编译器最终选择的布局是两份证据，应分别查看。**

## 最小完整示例：直接绘图

保持已安装 TileLang 的固定版本，仅补齐可视化依赖，避免为了安装 extra 无意升级 TileLang。

```python
import tilelang.language as T
from tilelang.tools import plot_layout

layout = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(layout, save_directory="./layout-output",
            name="transpose_4x4", formats="svg")
```

预期写入 `layout-output/transpose_4x4.svg`。这是主机侧映射图，不说明 kernel 实际采用该布局。

## 最小完整示例：查看推导

保存为 `inferred_layout.py`，执行时会编译，但不分配输入和运行 GPU kernel。

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="cuda", pass_configs={
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
})
def make_copy():
    @T.prim_func
    def copy_kernel(A: T.Tensor((8, 32), "float32"),
                    B: T.Tensor((8, 32), "float32")):
        with T.Kernel(1, threads=128):
            x = T.alloc_fragment((8, 32), "float32")
            T.copy(A, x)
            T.copy(x, B)
    return copy_kernel

if __name__ == "__main__":
    kernel = make_copy()
    print(kernel.get_kernel_source())
```

预期日志显示推导映射，并在工具报告的位置保存受支持的二维 fragment 图。编译需要可用的对应工具链；运行这个脚本不等同于已验证复制结果。

## 读图与限制

- Layout 默认 input view：格子是逻辑输入位置，标记是映射后的位置。
- Fragment：同时看线程 T 和线程局部索引 L，再对照访问表达式。
- 直接 Fragment 绘图要求受支持的二维输入及单值映射；其他形状可能仅能看文本。
- 非双射/稀疏映射可能覆盖图格或留下空洞；图不是布局合法性的证明。
- 小尺寸图适合排查；大图枚举成本高。最终仍检查生成代码并运行正确性测试。

## 来源与关联

- [docs/tools/layout_visualization.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/layout_visualization.md)
- [布局概念](layout-and-fragment.md) · [生成代码检查](../compilation/inspect-generated-code.md)
