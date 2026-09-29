# Selector 训练：范围与批量执行

2026-09-28：新增 R5b 校准小模型及其 train generation 已取消。2,048 个 query 的校准子集方案不再适用，不修改 `方案.md` 的多模态 evidence 补属性和 join discovery 机制。

## 当前需要执行什么

- 现有 PRIOR selector 已完成 7,029 个 Q–T 训练样本、两个交替 view、20 epochs，seed13/29 均有完成回执。当前使用的 seed13 checkpoint 是 `work/S2_COL_R2/PRIOR/checkpoints/13/selected.pt`，无需因代码提速重新训练并替换它。
- 约 15.7 小时的 train selector forward 和约 16.7 小时的 train generation 是为新增校准器准备数据；取消校准器后不再需要完成这两段。它们不是 selector MLP 自身的训练时间。
- selector 训练读取现成的冻结 9B reader 特征，不调用 generation。现有特征不重算，Qwen3.5-9B 不替换。dev/test 的论文效果评估仍保留。

## 实现

`column_r2_training.train_arm` 默认 `execution='batched'`，适用于 PRIOR、OO_CONTROL、FLAT_MIX；PVR evidence correction 分支保持原算法。旧的 `execution='scalar'` 可做受控对照。

`column_batching.score_records` 把每个 batch 的全部候选列拼接，合并大矩阵乘法，再按原样本边界拆回。MLP dropout 仍按每个样本的原形状与顺序调用。训练保留：

- 完整训练集、全部候选列、多 gold 的 set-mass loss、原 query/dataset 权重；
- batch32、最多 20 epochs、两个 view 交替、固定 shuffle 和 evidence 条件选择；
- AdamW、学习率、梯度裁剪、每轮 dev 选 checkpoint；
- 原 checkpoint 和数据 hash 校验。

FP32 矩阵合并可能改变低位浮点结果，不承诺新训练参数逐 bit 等于旧结果。最终层公共 bias 对所有列加同一常数，其理论 loss 梯度为 0；Adam 会放大其微小浮点残差，因此还须对比列概率、排名和 dev 指标，不能只比较参数 hash。

每轮 history 记录训练及验证耗时。新的训练可用 `--training-output` 分开保存 checkpoint、预测和指标，继续读取原 `--output` 的缓存。已完成训练的 MANIFEST 禁止覆盖。

## 早停和 loss 观察

PRIOR 默认启用早停。每处理 1,024 个 Q–T 样本（不是 1,024 个独立 query），记录这一段的平均 loss，并同时记录实际见过的 query 数。不同 batch 难度和交替 view 会影响训练 loss，因此不以某一段 loss 不下降直接停训。

每个完整 epoch 在全部 dev 上计算 MRR；默认第 10 轮开始累计耐心值，连续 5 次没有超过 0.0005（0.05 个百分点）的改善则早停。热身期间不积累失败次数，因此最早在第 14 轮结束时停。`selected.pt` 始终按原 MRR/H1 等规则保留真正的最佳 checkpoint，微小进步也能更新 checkpoint，`min_delta` 仅控制耐心值。停止后用该最佳 checkpoint 输出 dev 结果。

- `--early-stopping-patience 0` 关闭早停。
- `--min-epochs`、`--min-delta`、`--epochs` 调整热身、进步阈值和最大轮数。
- `--log-every-pairs` 调整 loss 记录间隔。
- OO_CONTROL、FLAT_MIX、PVR 默认仍跑固定预算，避免自动改变受控对比的训练步数；可以显式启用早停，但须在新实验中报告预算变化。

历史 dev 曲线回放见 `work/SELECTOR_EARLY_STOPPING_v1/REPLAY.json`：默认规则令 PRIOR seed29 在第 14 轮停止，仍选第 7 轮；seed13 保留至第 20 轮，仍选第 16 轮，两个 seed 的已选 dev 指标均无损失。这是历史回放，不能保证新训练都同样无损。更激进的第 6 轮开始计数方案会令 seed13 第 10 轮停止，MRR 下降 0.561 个百分点、H1 下降 0.851 个百分点，因此不设为默认。

## 仅在需要重训时运行

```bash
cd /tmp
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n MMDD python \
  /home/oycy/MMDD/src/run_stage2_columns_r2.py train --arm PRIOR --seed 13 \
  --output /home/oycy/MMDD/work/S2_COL_R2 \
  --training-output /home/oycy/MMDD/work/SELECTOR_TRAINING_BATCHED_v1
```

当前命令不进行 Stage1、reader 或 generation。更换 seed/arm 时仍要具备对应 reader 特征。不能用新源码冒充已冻结旧实验的源码；新训练回执记录实际执行方式及源码 hash。

## 速度与数值审计

`benchmark_selector_training.py` 只读取缓存，比较两种执行方式的 loss、梯度、参数更新、dropout RNG、训练及 dev 耗时、dev 排名；不生成新 reader 特征，不写入生产 checkpoint，不使用 test 标签。默认 256 对仅是工程探测，不是生产训练规模。

```bash
cd /tmp
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n MMDD python \
  /home/oycy/MMDD/src/benchmark_selector_training.py --train-pairs 7029 --epochs 20 \
  --out /tmp/selector_training_audit.json
```

训练计算加速、训练加验证加速和整个端到端实验耗时分别报告，不能互相替代。

2026-09-28 全量 7,029 对、两 view、20 epochs、seed13、CPU 4 线程对照（无早停）见 `work/SELECTOR_TRAINING_SPEED_FULL_v1/REPORT.json`：

| 项目 | scalar | batched |
| --- | ---: | ---: |
| 训练更新计算 | 138.83 秒 | 56.12 秒 |
| 训练 + 每轮 dev（不含缓存加载与 checkpoint I/O） | 151.66 秒 | 67.13 秒 |
| 所选 epoch | 16 | 18 |
| 所选 dev MRR | 0.921103 | 0.920244 |
| 所选 dev H1 | 0.854271 | 0.855528 |

训练计算约 2.47 倍，训练加验证约 2.26 倍。batched 的 MRR 少 0.086 个百分点，H1 多 0.126 个百分点；不是逐 bit 等价或全部排名一致。scalar 复现旧历史的第 16 轮及其 dev 指标。生产 checkpoint 的 hash 未变；对它执行两种前向时，675/675 dev 排名一致。已训练后的新模型轨迹存在累计浮点差异，不能拿前向一致性当作重训一致性。

检查命令（从隔离目录运行）：

```bash
cd /tmp
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/oycy/MMDD/src conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_column_batching.py \
  /home/oycy/MMDD/tests/test_stage2_columns_r2.py \
  /home/oycy/MMDD/tests/test_stage2_columns.py \
  /home/oycy/MMDD/tests/test_stage2_column_reporting.py \
  /home/oycy/MMDD/tests/test_lazy_selector.py \
  /home/oycy/MMDD/tests/test_r5b_fast.py \
  /home/oycy/MMDD/tests/test_stage2_verifier.py -q
```

结果：91 passed。包括早停热身/耐心值/禁用、最佳 checkpoint 恢复、完整样本/权重/条件 schedule、dropout RNG、loss/梯度/更新、停用校准器入口和原 selector/verifier 回归。
