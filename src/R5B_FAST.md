# R5b 校准实验已停止：保留 selector 推理审计

2026-09-28 用户取消新增校准小模型及其 train generation。`run_r5b_fast.py` 的命令行入口已停用，历史 helper 仅为已有审计和测试保留。当前工作见 [SELECTOR_TRAINING.md](SELECTOR_TRAINING.md)。原 artifacts、seal 和中断记录保留。

此前的 2,048 个 calibration query 方案不再适用；它不是 selector 训练规模。已有 selector 的训练规模为 7,029 个 Q–T 样本、两个交替 view、20 epochs。本轮没有启动全量 fast train generation，也没有完成新增校准器训练。

## 已验证的算法改动

全局列分数是 `log P(T) + log P(c|T)`，且 `log P(c|T) <= 0`。按表先验从高到低调用原 scalar reader；一旦未处理表的先验上界严格低于当前 max3-per-target 后第 10 个分支的分数，停止 reader。相等时继续计算，保留原 tie-breaking。表归一化分母始终包含完整 C50，后续 matching 也仍使用全部 C50 的完整列值域。

跳过的表仅记录上界证书，不捏造原始 logits。因此这里主动替代了旧协议的“交付所有 50 张表的完整 logits”要求；所有真正执行的分数、所选计划和跳过理由都保留。

2026-09-28 验证：完整 dev1198、test1166 和已有 train1009，共 3,373 个 query，用保存的完整 logits 回放后，**完整分支计划零差异**。平均需要约 7/50 张表。另有 8 个 train query 的 GPU 新鲜 exhaustive/scalar 与 lazy/scalar 对照：总计 400 次完整路径调用与 51 次 lazy 调用，耗时约 33.67 秒与 4.16 秒，约 8.10 倍加速；已执行列的最大 logit 差异为 0。详见 `work/R5b_FAST_AUDIT_v1/lazy_audit.json`。调用数比例不是端到端耗时保证。

## 当前执行范围

取消校准器后，不再为它执行 train Stage1、train selector forward、train generation、calibration features 或 fit。已有训练好的 selector 直接复用；只有需要新的 selector 训练时，才使用 `run_stage2_columns_r2.py train`。

上面的 lazy selector 是**推理优化**，不能用来剪掉训练候选列或改变训练 loss 分母。其 8.10 倍数据不是训练或端到端实验的加速倍数。本轮不继续 generation batch、校准子集、校准器学习曲线或替换 9B reader 的实验。

重复 selector 审计：

```bash
cd /tmp
conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/audit_r5b_lazy_selector.py \
  --out /tmp/r5b_lazy_audit.json --gpu-probe 8
```

测试从隔离目录运行：

```bash
cd /tmp
conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_lazy_selector.py \
  /home/oycy/MMDD/tests/test_calibration_cohort.py \
  /home/oycy/MMDD/tests/test_generation_batching.py \
  /home/oycy/MMDD/tests/test_r5b_fast.py \
  /home/oycy/MMDD/tests/test_stage2_verifier.py -q
```
