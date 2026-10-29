# JIT、目标架构与执行后端

> 关键词：target / arch / mcpu / execution_backend / CUDA / HIP。
> 适用：本批覆盖 NVIDIA CUDA 与 AMD ROCm；代码示例默认 CUDA；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**目标描述为哪张 GPU 生成什么代码；执行后端描述如何编译、装载和调用，二者需要一起记录。**

## 语义与约束

| 项目 | 应记录什么 | 不应混淆 |
| --- | --- | --- |
| TileLang | 安装路径、版本、源码 commit | 相同版本字符串不保证相同开发快照 |
| target | CUDA/HIP 及架构，如 sm_80 / gfx90a | 指定新架构不会让旧 GPU 获得新指令 |
| execution_backend | 实际选用的编译/调用实现 | 不仅是一个性能调参数字 |
| 工具链与运行时 | PyTorch、CUDA/ROCm、驱动、编译器 | 源码能生成不等于设备能装载执行 |

`auto` 适合便捷使用；复现时保存解析后的目标。CUDA 可用配置字典 `{"kind":"cuda","arch":"sm_80"}`；HIP 可用 `{"kind":"hip","mcpu":"gfx90a"}`，示例值必须替换为实际目标。

## 最小完整检查

以下脚本只检查环境，不安装、升级软件或编译 kernel。PyTorch 的 ROCm 构建也使用 `torch.cuda` 接口，因此打印 hip 版本帮助区分。

```python
import importlib.metadata
import torch
import tilelang

print("tilelang:", importlib.metadata.version("tilelang"))
print("module:", tilelang.__file__)
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("hip runtime:", torch.version.hip)
print("visible GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("properties:", torch.cuda.get_device_properties(0))
```

源码安装另外记录 `git rev-parse HEAD`；发布 wheel 的版本号无法自动证明等于本文开发快照。然后运行 [向量加](../api/kernel-and-buffers.md) 验证最小编译/调用链。

## 常见误用与检查

| 症状 | 检查顺序 |
| --- | --- |
| import 失败 | Python 环境、安装路径、FFI/依赖，不先修改 kernel |
| no kernel image | target 架构、生成产物与实际 GPU 的兼容性 |
| CUDA 代码在 AMD 上失败 | 使用 ROCm 语言接口与目标；替换 CUDA 专用同步/指令并重新验证 |
| 更新包后 API 不存在 | 核对源码快照和导入路径；不要混用 main 文档与旧 wheel |

本库不把 CUDA 示例标为已在 AMD 验证；两类 GPU 的迁移应保留独立编译与运行记录。

## 来源与关联

- [docs/get_started/targets.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/get_started/targets.md)
- [tilelang/jit/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/jit/__init__.py)
- [tilelang/language/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/__init__.py)
- [生成代码](inspect-generated-code.md) · [缓存与特化](specialization-and-cache.md)
