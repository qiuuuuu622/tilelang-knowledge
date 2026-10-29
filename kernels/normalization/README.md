# 归一化、mHC 与门控融合

先区分普通行归一化与模型专用融合。mHC 与 Engram 虽包含归约/归一化，输入输出语义都超出了 RMSNorm。

| 要比较的设计 | SGLang | vLLM | 首先核对 |
| --- | --- | --- | --- |
| pre 后处理 + RMSNorm | [SGLang](sglang-mhc-pre-rmsnorm.md) | [vLLM](vllm-mhc-pre-rmsnorm.md) | partial 布局、线程分工、carried pre_mix、BF16 舍入 |
| post + 下一次 pre 投影 | [SGLang](sglang-mhc-fused-post-pre.md) | [vLLM](vllm-mhc-fused-post-pre.md) | token/hidden 分派、split、后续 partial 合并 |

这些链接提供设计对照，不构成速度排名。比较时必须统一硬件、输入、误差容限、PDL、CUDA Graph、warmup，并计入前置/后续 kernel。

## 案例索引

共 10 篇。代码行数用于估计阅读成本；先读“什么时候选择”和“为什么这样设计”，匹配后再读代码。

| 案例 | 值得借鉴的设计 | 核心代码行数 |
| --- | --- | ---: |
| [SGLang mHC：pre mapping 与 RMSNorm 融合](sglang-mhc-pre-rmsnorm.md) | 消费 GEMM 的分段结果，生成 mHC mixing 系数和加权层输入，再在片上完成 RMSNorm。 | 267 |
| [SGLang mHC：post 与下一次 pre 投影融合](sglang-mhc-fused-post-pre.md) | 在生成新 residual 的同时计算下一阶段投影和平方和 partial，复用刚算出的值。 | 340 |
| [vLLM mHC：线程分工的 pre + RMSNorm](vllm-mhc-pre-rmsnorm.md) | 在线程组间分配 mixing 与 residual collapse 工作，融合 mHC pre 的后半段和 RMSNorm。 | 235 |
| [vLLM mHC：带适用边界的 post/pre 融合](vllm-mhc-fused-post-pre.md) | 将 post mapping 与下一次 pre 的 FMA 投影合并，并显式判断何时应回退到分开的实现。 | 192 |
| [vLLM mHC：两遍扫描的 head 融合](vllm-mhc-head.md) | 第一遍归约 RMS 和投影，第二遍应用 sigmoid 门控并合并 residual 通道，直接输出 head 输入。 | 108 |
| [TileLang 官方：RMSNorm 整行常驻与分段重读](tilelang-rmsnorm-resident-vs-chunked.md) | 用两个短实现比较：整行放入 fragment，或在同一 CTA 内分段累计后逆序重读。 | 60 |
| [TileKernels Engram：归约、门控与 residual 融合](tilekernels-engram-gate.md) | 归约归一化统计与门控分数，复用保存在片上的 hidden，再应用 value 更新；支持为反向保存统计。 | 199 |
| [TileKernels mHC：小矩阵 Sinkhorn 前后向](tilekernels-mhc-sinkhorn.md) | 把小 mixing 矩阵的多轮行列归一化放在片上，反向重放中间归一化状态。 | 154 |
| [TileOPs：按行宽与并行度选择 RMSNorm](tileops-rmsnorm-shape-dispatch.md) | 把 warp 内多行、整行常驻、通用 fragment 和两遍读取组织为有明确拒绝条件的候选实现。 | 583 |
| [TileOPs：residual add + RMSNorm + 动态量化](tileops-add-rmsnorm-quant.md) | 在同一计算链里完成 residual 相加、归一化、amax 与 INT8/FP8 量化，并保留量化前规定的舍入位置。 | 449 |

## 证据边界

全部条目已静态核查固定版本源码、保留完整选定函数及模块内依赖；未运行 GPU 编译、正确性或性能测试。上游测试入口不代表本仓库验证通过。复杂模型案例依赖对应上游工程，不能直接作为独立脚本运行。

[返回总索引](../README.md) · [来源、版本与收录边界](../SOURCES.md)
