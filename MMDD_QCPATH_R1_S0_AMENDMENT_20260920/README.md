# QCPATH-R1 S0 修订包

将 CONTINUE_PROMPT.md 作为消息，连同本包与原v1.0执行包交给执行模型。

AMENDMENT.zh-CN.md是本次有限修订，protocol.v1_1.json是已从原JSON生成的有效配置。原主规范仍适用，只有修订明示部分被覆盖。

本包没有修改服务器源码，没有运行GPU、重建标签或补算Qwen；它是继续执行指令。已在本地检查有效JSON可解析，并确认A/B模型与优化器超参数、性能门槛、原种子/namespace、检索配置及8作业上限未变。

历史划分依据：用户资料03_experiment_plans.md的R12第3节及R11数据协议，train_fit=11390、cal_fit=624、cal_check=616为历史身份线索；当前真实计数仍由服务器从独立划分与原始标注验证。
