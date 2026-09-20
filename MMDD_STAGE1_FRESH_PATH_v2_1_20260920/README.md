# MMDD Stage1 FRESH-PATH v2.1

完整fresh Stage1执行包；本版完整替代v2.0，不是需要叠加阅读的临时补丁。

**主线不变：**原dataset train/dev/test与原GT → fresh Teacher edge→path → fresh Student C1→条件化C2（SUP与KD）→ own ANN → T_PATH真实路径重排。

**新增对照：**独立训练T_QT；独立表格Student QT-SUP/QT-KD；没有adapter的旧式KD-NATIVE。原KD-EONLY保留，它读E但不读Q，不是无E组。

每seed 13个优化stage、6个Student终点、2个最终Teacher分支；最多2seed/26stage。公共T_EDGE与KD-C1前缀只训练一次。所有前缀均来自本轮fresh，不加载任何历史任务产物。

## 推荐阅读

先主文档 `EXPERIMENT_SPEC.zh-CN.md`，再 `protocol.json`、`CONTROL_MATRIX.md`、`EXECUTION_DAG.json`、`PARALLEL_EXECUTION.md` 与验收清单。将完整目录交给执行模型，使用 `CODEX_PROMPT.md` 启动。

## 双4090并行

只使用2×RTX4090（每卡24GB）。独立任务双卡并行；通过显存、正确性和吞吐测试后允许同卡2任务，全机最多4任务。并行不改变每作业模型/数据RNG、batch、精度或更新预算。正式时延测试独占GPU，并发吞吐另报。

## Fresh与历史成绩

历史B13/T0仅事后隔离参考，不能作为训练起点。KD-NATIVE+T_QT是明确的新fresh旧式结构工作点，不声称精确复现历史B13训练配方。QT-only不替代主方法；Path失败也不能改主表身份。

## 本包代码边界

`reference/`是CPU合成数学、结构、对照和调度合同参考；不是完整服务器训练程序。本次没有执行用户GPU训练或真实数据集评测。

运行参考测试：
```bash
cd reference
python -m pytest -q --disable-warnings
```

实际通过数量及命令见 `REFERENCE_TEST_RECEIPT.json`。源文件、配置和文档检查见 `PACKAGE_CONSISTENCY_CHECK.json`。执行端仍必须通过真实数据schema、Qwen提取、ANN、训练/评测和并行隔离验收。

不要只修改表格名称而不训练独立对照；不要混用旧协议JSON。
