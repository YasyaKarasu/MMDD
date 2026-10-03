# Stage-1 R3：让 Teacher 的证据支路真正起作用（任务说明）

日期：2026-10-03。基线：`work/stage1_entitables_r2_fullkd_s13`（协议 4.2.0）及其续训臂 `work/stage1_entitables_r2_fullkd_s13_teacher_sp`。
本轮**不跑完整管线**。先用 `src/train_teacher_chain.py` 从 init 训 TA→TB_CQET 并在 raw dev 池上评估（不含学生，回答架构问题），通过后再用 `src/continue_teacher_on_student_pool.py` 在学生池上续训（回答部署问题）。

## 1. R2 诊断结论（为什么要改 Teacher）

离线分析对象：R2 导出的 `logits.*.Real/Swap.jsonl.gz` 与 `funnels/*`（dev，1198 query，KD 学生池）。

| 事实 | 数值 |
|---|---|
| 池内 gold 目标 / 带证据包 / 包里含标注 witness | 997 / 577 / **108** |
| 带包目标是 gold 的概率 vs 无包 | 0.39% vs 1.28% |
| 续训 end：f0 / Real / Swap R@10 | 0.5002 / 0.4814 / 0.4853 |
| 续训后 LME(path) − f0 均值 | +1.10（run end 为 −2.75） |
| `f0 + 0.5·(LME_real − LME_swap)`（只保留证据内容的估计） | 0.4994 ≈ f0 |
| witness-oracle 包（gold 包里只用 witness 路径） | 0.5154（implicit 0.4723 vs f0 0.4420） |
| strict（direct top-100 漏掉的 gold）进 C150 后被 Teacher 放进 top-10 | KD 池 1/24，SUP 池 1/62，raw 池 86/141 |
| witness 第一跳 QE20 命中率：KD 学生 / SUP 学生 / raw | 28% / 79% / 73% |

机制：`aggregate_cqet` 对有包目标丢掉 f0、只用路径 LME；路径打分几乎不依赖证据（Real≈Swap），训练后只学到"有包→加分"的偏置，而有包目标恰恰更不可能是 gold，所以 Real < f0。KD 学生的 E 覆盖从 0.856 掉到 0.475 是把这种证据 logit 蒸给学生的直接后果。

## 2. 代码改动（默认值均等于 4.2 行为；88 个 Stage-1 测试通过）

- **`teacher.path_mode = "pairwise_residual"`**（`models.FreshPathTeacher`）：路径分 = f0(q,t) + path_head(q→e) + path_head(e→t)。两段局部关系走共享的 relation transformer + global MLP（方案.md 的 J(a→b)），`path_head` 末层零初始化，所以 warm start 时每条路径分 = f0，Real = f0；残差只能被证据相关的损失推动。`triplet` 为旧模式，保留做消融。
- **`teacher.TB.path_loss_scope = "bagged"`**：聚合路径的 rank-mass 只在有包目标之间做，"有无包"本身不再带监督。
- **`teacher.TB.witness_target_weight`**：TB 记录带 `qet_lists`（与 TA 同款：对每个 witness w，候选 = 学生空间 ET128(w) ∪ C150 ∪ G），损失是 (q,w,t) 路径分在候选目标上的 rank-mass。
- **`teacher.TB.support_competitors`**（默认 8）：witness-vs-competitor 对比的竞争者数。
- **`retrieval.formal_hops = "exact"`**：GPU 精确 top-k，不建 HNSW；每 query 91 ms（原 ~200 ms + 每趟 6 min 建索引）。注意 exact 池与 HNSW 池不逐位相同（HNSW 第一跳 overlap ≈0.95）。
- 新入口 `src/train_teacher_chain.py`：从 init 训 TA→TB_CQET，把 TA 的 `qet_lists` 并进 TB 记录，在 raw dev 池上评 TA_epoch*/TB_half/TB_end 的 f0/Real/Swap、strict 漏斗和配对 bootstrap（含对 R2 参考链的对比）。
- 效率：Teacher 打分核心改为 token 池单次 gather（TB 步 5.4→2.8 s，loss 逐位相同）；D1 贪心批量化（0 mismatch，2.3×）；推理 chunk 1024；`student.C2.prepaths_audit=false` 只写 retained 路径。

## 3. 实验设计：先从 init 训 Teacher（不含学生），再在学生池上续训

### 3.1 第一步：从 init 的 TA → TB_CQET 链（架构 A/B，入口 `src/train_teacher_chain.py`）

为什么先做这一步：TB 冻结 `adapters/poolers/globals`，证据 token 支路里真正"看证据"的编码器只在 TA 训练；TA 记录每个 witness 一条 QET 列表，是最干净的证据监督；raw 池里 gold 包含 witness 的比例（28%）和第一跳命中率（73%）都远好于 KD 学生池（19% / 28%）。对照臂就是 R2 自己的 TA→TB_CQET 链：同一套 `TA.jsonl.gz` / `TB_SHARED.jsonl.gz`、同一 seed、同一 raw dev 池（raw dev：f0 0.4521 / Real 0.4479）。脚本把 TA 的 `qet_lists` 按 query 并进 TB 记录，不需要重新挖掘；support 竞争者沿用记录里的 8 个。

```bash
R=work/stage1_entitables_r2_fullkd_s13
P=/home/oycy/miniconda3/envs/MMDD/bin/python
# A 主臂：残差 + 仅有包目标的路径损失 + witness 目标列表 + support 权重 1.0
setsid nohup $P src/train_teacher_chain.py --run-root $R --out-dir work/stage1_r3_chain_residual_A --gpu 0 \
  --path-mode pairwise_residual --path-loss-scope bagged --witness-target-weight 0.5 --support-weight 1.0 \
  > work/stage1_r3_chain_residual_A.log 2>&1 &
# B 对照：残差结构 + 4.2 的损失（scope all、support 0.2、无 witness 列表）——隔离"结构"与"损失"
setsid nohup $P src/train_teacher_chain.py --run-root $R --out-dir work/stage1_r3_chain_residual_B --gpu 1 \
  --path-mode pairwise_residual > work/stage1_r3_chain_residual_B.log 2>&1 &
# C 对照（A/B 之后）：triplet 结构 + A 的损失——隔离"结构"的贡献
#   同 A 但 --path-mode triplet
# 参考臂：R2 的 TA→TB_CQET（triplet + 4.2 损失），结果已在 $R/seed13/eval/dev/SUMMARY.json，脚本会自动做 TB_end 对它的配对 bootstrap
```

单臂预计：TA 2 epoch 约 40 min、TB 1 epoch 约 30 min（新打分核心），评估 5 个 checkpoint × 1198 query 约 20 min。冒烟：`--limit 16`。
产物：`<out>/TA/{init,epoch1,epoch2}.pt`、`<out>/TB_CQET/{init,half,end}.pt`、`<out>/{TA,TB_CQET}.train.jsonl`、`<out>/eval/dev/raw/`（rankings/logits/funnels）、`<out>/eval/dev/SUMMARY.json`。

### 3.2 第二步：在学生池上续训（部署问题，入口 `src/continue_teacher_on_student_pool.py`）

只有第一步在 raw 池上确认证据支路有效之后才做。起点用第一步主臂的 `TB_CQET/end.pt`（或 R2 续训臂的 `end.pt` 做 warm start），在 KD 学生池列表上训 1 epoch，评估在学生 dev 池上。这一步回答的是"在学生实际召回的池子上残差能否在 f0 之上再加分"。

```bash
INIT=work/stage1_r3_chain_residual_A/TB_CQET/end.pt
setsid nohup $P src/continue_teacher_on_student_pool.py --run-root $R --out-dir work/stage1_r3_teacher_sp_A \
  --gpu 0 --init-checkpoint $INIT --path-mode pairwise_residual --path-loss-scope bagged \
  --witness-target-weight 0.5 --support-weight 1.0 --support-competitors 16 --search exact \
  > work/stage1_r3_teacher_sp_A.log 2>&1 &
```

预计挖掘约 20 min、训练约 1.3 h、评估约 20 min；`--support-competitors 32` 时峰值显存 23.8 GiB（接近 24 GiB，OOM 梯子会接住但变慢），先用 16。

## 4. 验收指标

第一步（`<out>/eval/dev/SUMMARY.json`，raw dev 池，脚本已算好配对 bootstrap）：

1. `TB_end.Real_minus_f0`：主臂应 > 0 且 CI 不跨 0。R2 参考链是 −0.4pp（0.4479 − 0.4521）；TA 阶段参考链 Real 比 f0 低 8.6pp（0.3177 vs 0.4036），残差版从 init 起应始终 ≥ f0 附近。
2. `TB_end.Real_minus_Swap`（E-content）：应显著 > 0；参考链在 128-query 探针上 Real 0.4531 vs Swap 0.4505。
3. `strict_pairs.TB_end.teacher_top10`（direct top-100 漏掉的 gold 进 C150 后被放进 top-10 的对数）：参考链 86/141，要看能否明显上升。
4. `TB_end.f0_minus_reference.f0`：f0 不应退化（直接损失仍在）。
5. 离线复算（`/tmp/mmdd_r3/logits_analysis*.py` 同款，改路径指向新目录）：witness-oracle 包 R@10、Real 前十的有包占比不应系统性高于 f0 前十。

第二步（学生 dev 池）：同前 1–4，另加 `end.Real − 学生 Direct_ANN_R10` 和 raw 池 f0 下降 ≤ 1pp。

## 5. 之后再做的事（本轮不做）

- Teacher 证据支路验收通过后，用它重做 `teacher_logits_cache`（含 evidence KD）再蒸学生，看学生 E 覆盖能否从 0.475 回到 ≥ 0.85。
- 第一跳本身：witness 在 KD 学生的 QE20 里只有 28%，SUP 学生 79%。证据 KD 修好前，Teacher 训练用 SUP 学生池（`--student SUP`）会给出更多含 witness 的包。
- 如果残差臂的 Real−Swap 仍 ≈ 0：说明 3 层/512 的 token 支路学不到行级桥接，再考虑显式 bridge 特征（query 行 × 证据的 row_support 统计、证据 slot × 目标行的匹配）送入 path_head。

Gemini 报告里的四个方向：方向 1 的"换成 Qwen2-VL 做判别底座"不必要——瓶颈是训练信号与聚合规则而非容量（置零 token 支路只掉 1.6pp）；方向 2（直接推进 Stage-2）会把一个 Real<f0 的证据分数带进下游；方向 3（闭环再蒸）在证据支路修好前只会再次压垮学生的 E 覆盖；方向 4（显式/隐式路由）用的是 query 标签本身，不能作为系统方法。
