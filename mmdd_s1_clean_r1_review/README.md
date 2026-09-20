# MMDD CLEAN-R1 独立复核包

- INDEPENDENT_REVIEW.zh-CN.md：主分析，区分数值事实、代码缺陷、机制推断和验证边界。
- SOURCE_LOCATIONS.md：上传源码的精确文件/行号/片段和文件hash。
- RECOMPUTED.json：1198-query同池指标、六轮复算、候选覆盖、来源分层、paired bootstrap、固定panel Top-k。
- SOURCE_PROBES.json / STUDENT_PROBES.json：对上传代码运行的合成探针结果。
- PREFIX_STRATA.json：Teacher pair-accuracy的候选来源前缀分层，前缀仅用于分析，不作为模型输入。
- REPAIR_AND_REPLAY.md：给执行代理的修复与验收合同；先验收，后新run重跑。
- recompute.py / probe_source.py / probe_student.py：复算与探针脚本。

这些脚本不修改原实验文件、不运行Qwen、不在真实数据上训练。probe_source.py仅在临时目录执行一个模拟的难例刷新，以记录实际被调用的模式与候选。

复跑：先解压用户的原始mmdd_s1_audit_20260918.tar.gz，再设置

```bash
export MMDD_AUDIT_ROOT=/绝对路径/mmdd_s1_audit_20260918
OPENBLAS_NUM_THREADS=1 python recompute.py
OPENBLAS_NUM_THREADS=1 python probe_source.py
OPENBLAS_NUM_THREADS=1 python probe_student.py
```

需要numpy；后两个需要本机torch。输出写到脚本所在目录。脚本没有访问原19GiB缓存或服务器GPU；不能把成功复算排名与合成探针，解释为已经独立重跑真实Teacher前向。
