# 异步错误怎样保留并缩成可复现用例

> 关键词：race / barrier / phase / T.print / AutoDD / deterministic failure。
> 适用：GPU；显式异步协议需对应架构，AutoDD 会执行候选程序；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先固定一个可靠的失败判据，再缩小用例；每一次缩小都必须保留原问题。**

## 排查顺序

1. 固定 seed、输入、形状、GPU、编译选项，重复运行，记录失败比例而非只保留一次 PASS。
2. 为每个共享缓冲列出“生产完成 → 消费开始 → 消费完成 → 下一轮覆写”，核对同步范围与参与线程。
3. 暂时退回 serial 或减少 stage；若错误消失，只是缩小怀疑范围，还不能证明哪个 barrier 有错。
4. 覆盖短循环、奇偶轮、尾块与不同线程配置；改变调度可能暴露旧错误。
5. `T.print` 可辅助定位，但插桩会改变调度和时间；打印后错误消失不等于修复。

## 最小复现与缩减

把失败保存成独立 `repro.py`，包含输入构造、kernel、调用与断言。结果错误时由断言产生独特的 `RESULT_MISMATCH`；编译错误则选择原错误中特定的短语。先直接运行确认稳定失败，再执行：

```bash
python -m tilelang.autodd repro.py --err-msg "RESULT_MISMATCH" --backend subproc -j 1 -o minimized.py
python minimized.py
```

命令中的错误短语必须真的由失败判据产生，不能无条件打印；上述文件是用户的复现输入，不是仓库中已有文件。缩减需要运行 kernel，因此要在复现所需 GPU 环境执行。

## 工具边界

AutoDD 按 stdout/stderr 中的大小写敏感子串保留候选，不要求非零退出码。太宽泛的 `error`、`failed` 可能保留无关失败。`subproc` 每次使用新进程，适合排除残留状态；GPU 候选先串行运行，避免互相争抢资源。

对偶发竞争，不直接用单次 AutoDD 结果归纳：先让判据重复足够次数，保留原始失败概率与完整脚本；必要时手工缩减。随机失败率下降不是语义修复证明。

## 完成条件

缩减版仍重现同一错误；修复在缩减版和原版均通过；回归覆盖多轮缓冲复用与边界。性能测量使用移除插桩的版本，不能拿调试运行时间作为优化结果。

## 来源与关联

- [docs/tools/autodd.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tools/autodd.md)
- [docs/tutorials/debug_tools_for_tilelang.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/tutorials/debug_tools_for_tilelang.md)
- [docs/programming_guides/software_pipeline.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/software_pipeline.md)
- [同步契约](../semantics/synchronization.md) · [流水模式及反例](../../patterns/pipelining/budget-pipeline-resources.md)
