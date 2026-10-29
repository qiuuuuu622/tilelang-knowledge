# TileLang 高性能算子案例

面向选型与改写的案例库。每篇一个 Markdown：**做什么 → 何时值得选 → 为什么这样设计 → 完整核心代码 → 固定版本来源与验证入口**。

本轮整理 **33 篇**，来自 **5 个上游仓库**。这里的“高性能”指值得分析的优化实现，不是已经在本仓库测得最快。全部是源码静态核查；没有 GPU 实测记录。

## 按任务找案例

| 分类 | 篇数 | 首先区分 |
| --- | ---: | --- |
| [Attention 与稀疏索引](attention/README.md) | 11 | 最终 attention / 索引评分 / pooling Top-k；dense / sparse；prefill / decode |
| [GEMM / GEMV](gemm/README.md) | 5 | 大 M / 小 M；普通布局 / 显式指令；全精度 / 打包权重 |
| [归一化、mHC 与门控融合](normalization/README.md) | 10 | 普通 RMSNorm / mHC / Engram；整行常驻 / 重读 / 融合 |
| [量化与激活融合](quantization/README.md) | 4 | 分组粒度、scale 格式、舍入与 packing；量化前是否还有激活 |
| [Softmax 与 Top-k](reduction/README.md) | 3 | 小 k 选择 / 整行 softmax；短行 / 超长行 |

## 按问题找设计

| 观察到的问题或需求 | 先读 | 实验要区分什么 |
| --- | --- | --- |
| decode 的 CTA 太少 | [MLA split-K](attention/tilelang-mla-decode.md)、[DSA partial/combine](attention/sglang-dsa-split-decode.md) | 增加并行度是否足以抵消合并成本 |
| 稀疏 KV 装载和计算不能重叠 | [DSA 基础版](attention/sglang-dsa-sparse-baseline.md)、[手动流水版](attention/sglang-dsa-warp-specialized.md) | 等待是否减少，shared/register 是否限制占用率 |
| pooling、选择产生大中间矩阵 | [MiniCPM pooling + Top-k](attention/sglang-minicpm-pooling-topk.md) | 节省的写回是否值得重算 QK |
| GEMM 怀疑调度/缓存问题 | [persistent 对照](gemm/tilelang-persistent-gemm.md) | 排除 tile 与精度变化后，调度顺序是否有收益 |
| 权重带宽高、M 很小 | [INT4 GEMV](gemm/tilelang-int4-dequant-gemv.md) | 严格精度合格后，解包成本与带宽节省谁占主导 |
| 宽行寄存器压力高 | [RMSNorm 分派](normalization/tileops-rmsnorm-shape-dispatch.md)、[softmax 流式](reduction/tileops-softmax-resident-streaming.md) | spill/占用率改善能否弥补重读 |
| mHC 多个小 kernel 与中间读写 | [SGLang/vLLM 对照](normalization/README.md) | 融合后全链路时间与舍入是否可接受 |
| 归一化后立即量化 | [add + RMSNorm + quant](normalization/tileops-add-rmsnorm-quant.md) | 两个归约、舍入和 scale 契约都能否保持 |
| SwiGLU 后立即量化 | [SwiGLU + quant](quantization/tilekernels-swiglu-quant.md) | clamp/路由权重与量化顺序是否一致 |
| 少量 expert 的 Top-k | [重复 argmax](reduction/tilekernels-topk-gate.md) | k 增大到哪里，排序开始更划算 |

## Agent 阅读与复用

1. 从分类索引按计算语义、shape、dtype、硬件和布局筛选；不要只按算子名选择。
2. 先读候选的“适用/不适用”和三条设计取舍，选出少量有理由比较的实现。
3. 再读核心代码。工厂、完整计算链和框架入口的边界都在代码前说明；保留导入不代表依赖已安装。
4. 按“下一步验证”建立能区分假设的实验。比较完整调用链，保持相同精度、输入与测量设置。
5. 将本地环境、正确性结果和测量记录放到 [workbench](../workbench/)，有证据后再更新结论。先发现瓶颈，再选择模式。

来源固定在各自 commit，**不是一套统一兼容的运行环境**。执行时使用各上游自己的依赖约束，并另行记录 TileLang、CUDA/ROCm、驱动与 GPU 型号。[来源登记与收录边界](SOURCES.md)列出本轮范围和去重理由。

## 单篇模板

下面是结构约定；案例正文不增加独立的代码、测试、结果目录。

````markdown
# 来源项目：设计主题

一句话说明计算、输出和优化切入点。

## 什么时候选择

- 适用：计算语义、输入形状/dtype/布局、硬件与调用前提。
- 不适用：会错误、无法复用或预期不划算的条件。
- 性能依据：实测条件与来源；没有测量就写机制分析。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 可定位的问题 | 对应具体变量/循环/布局的选择 | 收益假设、资源成本及失败边界 |

下一步验证：用什么正确性检查或对照实验区分这些假设。

## 核心完整代码

说明入口和依赖边界。保留完整计算、同步、尾部处理与必要辅助函数，不用省略号代替代码。

```python
# 完整核心实现及必要导入。
```

## 来源与验证

固定 commit、文件/符号、提取或修改说明、参考/测试入口、许可证、核查日期。
````
