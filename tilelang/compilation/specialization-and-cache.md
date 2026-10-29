# 如何确认测到的是目标特化实现

> 关键词：JIT cache / autotune cache / TILELANG_DISABLE_CACHE / hashable parameters。
> 适用：GPU 编译与实验复现；缓存细节按固定版本；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**性能记录应绑定源码、特化参数、目标、编译选项与输入，而不只记录一个函数名。**

## 语义与约束

- factory 的 N、tile、dtype 等编译期参数可能产生不同特化；运行时 Tensor 不应被当作普通可哈希配置参数。
- 编译期参数及其默认值需要可哈希，且影响特化的值、hash/equality 语义应保持稳定。
- JIT 缓存、autotune 最优配置缓存和原始实验记录是不同对象。找到 best config 不代表保留了每轮失败原因。
- 不把模块全局变量的隐式变化当作可靠配置更新机制。影响代码的因素应显式进入配置，并检查生成结果。

## 最小检查

保存 [vector_add.py](../api/kernel-and-buffers.md)，用新进程运行下面命令以排除缓存命中对 lowering 检查的干扰：

```bash
TILELANG_DISABLE_CACHE=1 TL_LOWER_TRACE=1 python vector_add.py
```

要隔离一次实验的磁盘缓存而不删除其他工作，可在 shell 中创建专用目录：

```bash
experiment_cache=$(mktemp -d)
TILELANG_CACHE_DIR="$experiment_cache" python vector_add.py
```

预期仍通过正确性检查；是否冷编译从编译日志与 trace 判断。记录此次目录，但不要把缓存文件当作长期可移植的知识条目。

## 应保存的最小记录

| 身份 | 内容 |
| --- | --- |
| 代码 | 本仓库与 TileLang commit；kernel 源码或其可追溯 hash |
| 配置 | shape、dtype、stride、tile、threads、stages、pass 配置 |
| 环境 | GPU、驱动、工具链、执行后端 |
| 阶段 | 冷编译时间 / 首次调用 / 预热后的执行时间分别记录 |
| 结果 | 正确性、容差、原始时间样本、失败状态 |

## 常见误用与检查

| 症状 | 下一步 |
| --- | --- |
| 改参数后源码没变 | 确认该参数确实参与构建；新进程禁用缓存检查 |
| trace 空或缺少 pass | 确认是否命中缓存，不能直接认为 pass 未执行 |
| autotune 命中但没有 trial 历史 | 保留 cache-hit 身份；需新 trial 时另开实验，不伪造旧记录 |
| 热运行很快，首次很慢 | 分离编译、加载、预热与执行，不混成单一 latency |

## 来源与关联

- [tilelang/jit/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/jit/__init__.py)
- [tilelang/cache/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/cache/__init__.py)
- [docs/programming_guides/autotuning.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/autotuning.md)
- [测量方法](../tuning/measurement.md) · [autotune](../tuning/autotuning.md)
