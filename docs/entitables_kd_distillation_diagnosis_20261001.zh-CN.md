# EntiTables 上 Teacher→Student 蒸馏无效的诊断与可行方案（2026-10-01）

对象：`work/mmdd_stage1_v4_1_correctness_locked/lrsweep/` 下的 EntiTables C2 学习率 sweep（`LR0`、`R10` 两臂），
以及它所依赖的预注册 run `seed13/`。AbeBooks 的 `runs/fresh_R10` 不在本文范围内。

## 0. 结论

**不是学生结构到了上限，是学生从来没有被真正训练过，而且管线里的 KD 配置即使训练正常也等价于 SUP。**

三个独立的原因叠在一起，任何一个都足以让「KD − SUP = 0」：

1. **学习率太小**：预注册 `P_lr=1e-6 / R_lr=1e-5`，C1 的 479 步和 C2 的 198 步里 loss 纹丝不动，
   参数范数只变了 1e-5 的相对量。KD 和 SUP 的 endpoint 实际上是同一个模型。
2. **打分尺度错配**：学生的分数是 cos 量级（std ≈ 0.06，范围 ≈ 0.5），teacher 的 logit std ≈ 6、范围 ≈ 39。
   温度 τ=1 下学生的 softmax 完全平坦，KL 的梯度退化成「把 teacher 的 top-1 推高」的硬标签梯度，
   与 SUP 梯度几乎重合——teacher 对其余 750 个候选的排序信息（真正的 dark knowledge）梯度权重为零。
3. **×10 学习率臂被训崩了**，但 sweep 的召回测量脚本有 bug，没人发现。R10 的 dev 直接 R@10 是 **0.0025**（LR0 是 0.2723）。
   bundle MANIFEST 里「R10 与 LR0 召回曲线逐位相同、评估路径吸收了扰动」是脚本读错文件得出的。

用一个合理的配方（随机负样本 + logit 尺度 20 + 学习率 ×100）在同一份 C2 数据上重训 direct-only 学生，
一个 epoch（198 步，GPU 上 8 秒）：

| 1 epoch，3 种子均值 ± std | dev R@10 | test R@10 | dev C150 池内 R@10 |
|---|---|---|---|
| 当前 endpoint（C1 末端 / LR0） | 0.2673 / 0.2723 | 0.2682 / 0.2747 | 0.2673 |
| SUP（合理配方） | 0.4327 ± 0.013 | 0.4137 ± 0.012 | 0.4449 |
| SUP + 0.3·KD，τ=1（管线配置） | 0.4280 ± 0.015 | 0.4097 ± 0.010 | 0.4474 |
| **KD-only，τ=10** | **0.4773 ± 0.008** | **0.4546 ± 0.007** | 0.4648 |
| SUP + 1.0·KD，τ=10 | 0.4607 ± 0.007 | 0.4420 ± 0.011 | **0.4713** |
| （参照）teacher f0 在同一 C150 池上 | — | — | 0.4633 |

KD 在温度对齐后比 SUP 高 **+4.5pp（dev）/ +4.1pp（test）**，远超种子噪声；学生在 teacher 所重排的同一个 150 候选池内达到了 teacher 直接打分（f0）的水平。
3 个 epoch 的 KD τ=10 峰值到过 dev R@10 = 0.4896。学生这个双线性结构的上限至少在这里，离目前的 0.27 很远。

## 1. 检查了什么

- `lrsweep/README.md`、`ADAPTATION.md`、`LR0/`、`R10/` 下的全部产物与训练日志；`scripts/` 下的训练与测量脚本。
- `seed13/eval/{dev,test}/SUMMARY.json`：预注册评估的 raw / native_sup / native_kd / qt_sup 四臂。
- `src/mmdd_cqet_v4_1/{train,losses,models}.py`：C2 训练循环、`rank_mass_loss`、`list_kl_divergence`、`NativeStudent`、`FreshPathTeacher`。
- `seed13/teacher_logits_cache/scores.pt`：冻结的 TB_CQET teacher logits（12,630 个训练 query）。
- 自己写的三个探针（`lrsweep/probe_direct_kd/`）：endpoint 精确召回、R10 崩溃轨迹、direct-only 重训探针。

## 2. 诊断证据链

### 2.1 预注册配方下，学生没在学

C2 训练日志（`LR0/NATIVE_SUP.train.jsonl`）：

| step | direct_sup_loss | evidence_sup_loss | P 范数 | R 范数 |
|---|---|---|---|---|
| 1 | 6.464 | 6.368 | 55.4617 | 71.7012 |
| 100 | 6.360 | 6.266 | — | — |
| 198 | 6.465 | 6.523 | 55.4624 | 71.7033 |

- `direct_sup_loss ≈ 6.46 ≈ log(753)`，753 正是每条 C2 记录的平均候选数。`rank_mass_loss = LSE(all) − LSE(pos)`
  在学生分数平坦时恰好等于 log(N/|P|)，也就是说学生对候选的 softmax 是**均匀分布**，gold 的概率 ≈ 1/N。
- 参考值：3 份 PCA 基的 Frobenius 范数 = 55.43，5 个单位阵 = 71.55。学生整个 C1+C2 训完还停在初始化点附近。
- C1 同样：479 步 loss 3.72 → 3.72。
- 后果：KD 与 SUP endpoint 是同一个模型。精确重算（`probe_direct_kd/results/endpoint_direct_recall.json`）：
  dev 直接 R@10 LR0_SUP 0.2723 / LR0_KD 0.2714，903/1198 个 query 的 top-10 **逐位相同**，top-150 集合重叠 99.6%。
  这两个数与 `SUMMARY.json` 的 `Direct_ANN_R10`（0.2723 / 0.2714）一致，确认 LR0 就是预注册 endpoint。

### 2.2 尺度错配：τ=1 的 KL 退化成硬标签

Teacher logits 缓存上的统计（12,630 个训练 query，direct 列表）：

| 量 | 均值 | p10 | p50 | p90 |
|---|---|---|---|---|
| 候选数 | 753 | 397 | 734 | 1133 |
| logit 范围（max − min） | 38.8 | 32.3 | 38.9 | 45.4 |
| softmax 熵（nat） | 1.10 | 0.05 | 0.95 | 2.41 |
| softmax top-1 概率 | 0.68 | 0.29 | 0.71 | 0.99 |

学生分数：全库 std ≈ 0.06，top-150 内范围 ≈ 0.15。

推导：学生均匀时 KL(T‖S) = log N − H(T) ≈ 6.62 − 1.10 = 5.5，与日志里 `direct_kd_loss ≈ 5.2–5.5` **完全吻合**。
此时 ∂KL/∂s_i = p_S(i) − p_T(i) ≈ 1/N − p_T(i)：teacher top-1 得到 −0.68 的推力，其余所有候选都是 +1/N——
teacher 对非 top-1 候选之间的相对排序对梯度没有贡献。而 teacher 的 top-1 大多数时候就是 gold，
所以 0.3·KD ≈ 把 SUP 的权重从 1.0 调成 1.3，**信息上等价于 SUP**。

探针直接验证了这一点（第 3 节）：训练正常时，`SUP + 0.3·KD, τ=1` = 0.4280，`SUP` = 0.4327，无差别；
而 `KD-only` 随温度：τ=1 → 0.371，τ=5 → 0.441，τ=10 → 0.468，τ=20 → 0.467。

### 2.3 ×10 学习率臂：秩一漂移导致崩溃

`probe_direct_kd/results/collapse_trace.jsonl`（R10 / NATIVE_SUP，dev 直接 R@10 与 `R_QT − I` 的奇异值）：

| 快照 | dev R@10 | ‖R−I‖_F | σ₁ | σ₂ | meanU·ΔR·meanU / ‖meanU‖² |
|---|---|---|---|---|---|
| 0% | 0.2673 | 0.57 | 0.20 | 0.16 | 0.001 |
| 25% | **0.2796** | 1.60 | 0.62 | 0.47 | 0.021 |
| 50% | 0.2694 | 2.69 | 1.42 | 0.81 | 0.187 |
| 75% | 0.0303 | 4.34 | 3.05 | 1.17 | 0.902 |
| 100% | 0.0025 | 6.49 | 5.22 | 1.49 | 1.886 |

σ₁ 一枝独秀、σ₂ 几乎不动：ΔR 是一个**秩一**项，方向就是 target 嵌入的均值方向。学到的是一个与 query 无关的
「通用表偏置」，对所有 query 都返回同一批表。机制：C2 的负样本只来自候选池（池里都是和 query 相近的表），没有任何
随机/全局负样本约束池外对象；加上 softmax 平坦时梯度对所有 query 高度一致、Adam 逐元素符号归一化又把一个秩一梯度
放大成每步谱范数 ≈ lr × 1024 的稠密更新。这与 `v3.1 Path loss 半梯度漂移` 记录的 teacher 公共分量漂移是同一个机制。

值得注意：25% 快照时两臂都在**上涨**（SUP 0.2796，KD 0.2805），说明信号可学，只是随后被漂移吞掉。

探针里的 `CTRL_recipe_like`（尺度 1、无随机负样本、τ=1、权重 0.3、lr 1e-4/1e-5）精确复现了这条轨迹：
0.267 → 0.282 → 0.252 → 0.026 → 0.003，σ₁=5.4 对 σ₂=1.7。

### 2.4 sweep 自己的测量有三处 bug

1. `scripts/measure_k_curve.py` 和 `scripts/bootstrap_kd_gap.py` 接收 `--arm`，但 rankings 路径写死为
   `seed13/eval/<split>/native_sup|native_kd/rankings.TB_CQET.Real.jsonl.gz`。`LR0/` 与 `R10/` 下的
   `K_CURVE_*.json`、`BOOTSTRAP_*.json` 是同一份文件，都是预注册 run 的数字。LR0 恰好等于预注册 endpoint 所以
   碰巧是对的；R10 的全错。
2. `scripts/measure_c2_entitables.py` 用 `set(gt[query])` 取 gold，但 `load_split_gt` 返回的是
   `{"G", "implicit_G", "kind", "source_group", "W"}` 这样的 dict，应为 `gt[query]["G"]`。`KD_EFFECT_*.json` 里
   `coverage_sup/kd` 恒为 0.0。
3. 因此 R10 的召回从未被测过。`entitables_lr_sweep_bundle.tar.zst` MANIFEST 中
   「Raising the learning rate x10 … the SUP/KD recall curve is identical to LR0 at every K」与
   「it is the evaluation path absorbing sub-rank perturbations」两条结论不成立。

### 2.5 Teacher 比学生强在哪

- `TB_CQET.f0`（teacher 的直接打分，不走证据路径）在 native_sup 的 C150 池上 R@10 = 0.4633；`Real`（带证据）0.4714。
  Teacher 的优势几乎全在 f0，证据路径只贡献 +0.8pp（test 上为负）。
- 学生直接检索全库 R@10 = 0.27，在同一 C150 池内也是 0.27。所以 19pp 的差距是 **cross-encoder f0 vs 双线性型** 的差距。
- Teacher f0 = `scoring_head( Transformer([REL, q_tokens, SEP, t_tokens])[0] + MLP([g_q, g_t, g_q⊙g_t, |g_q−g_t|, pair, 0…]) )`，
  是 (z_q, z_t) 的非线性函数；学生是 `(z_q−μ)ᵀ Pᵀ R P (z_t−μ)`。结构差距真实存在，但第 3 节表明它目前不是瓶颈。

## 3. 上限还是策略：direct-only 重训探针

### 3.1 设计

`lrsweep/probe_direct_kd/probe_kd.py`。只训 `P.table` 和 `R.QT`（direct 关系），不碰证据分支；数据、初始化与 sweep 完全相同：

- 训练列表：`seed13/training_records/C2_SHARED.jsonl.gz`（12,630 个训练 query，每条 ≈ 753 个候选，≈ 1 个 gold），
  teacher 分数来自 `seed13/teacher_logits_cache/scores.pt`；初始化 = `NATIVE_C1_SUP/snapshot_frac100.pt`（sweep 的父 checkpoint）。
- 相对预注册配方的三处改动，各自可开关：
  1. 每个 query 追加 256 个从 22,886 个合法 target 中均匀抽样的**随机负样本**（只进 SUP 项；KD 项仍只覆盖池内）；
  2. 学生分数乘固定 **logit 尺度 20**（cos std 0.06 → 1.2）；
  3. KD 的 teacher 端除以**温度 τ**（std 6 / τ）。
- 学习率 R 1e-3 / P 1e-4（预注册的 100 倍），AdamW，梯度裁剪 1.0，batch 64，1 epoch = 198 步，无 anchor。
- 评估：dev/test 全库精确直接召回 R@{10,50,150}；dev 在 native_sup 的冻结 C150 池内的 R@10（与 teacher f0 的 0.4633 可比）；
  `σ₁/σ₂(R_QT − I)` 作漂移监控。训练 / dev / test 三个 query 集合两两不相交（已核对）。
- 整个向量化训练 198 步只需 8 秒（管线的逐 query 循环要 20 分钟）。

### 3.2 结果

单种子（seed 13）全表，1 epoch 除非注明：

| 臂 | dev R@10 | dev R@50 | dev R@150 | dev C150 R@10 | test R@10 | test R@150 | σ₁ | σ₂ |
|---|---|---|---|---|---|---|---|---|
| 初始化（C1 末端） | 0.2673 | 0.4341 | 0.5809 | 0.2673 | 0.2682 | 0.5893 | 0.20 | 0.16 |
| SUP | 0.4181 | 0.6704 | 0.8173 | 0.4388 | 0.4004 | 0.8398 | 4.42 | 4.35 |
| KD-only τ=1 | 0.3708 | 0.5912 | 0.7403 | 0.4455 | — | — | 5.16 | 4.74 |
| KD-only τ=5 | 0.4412 | 0.6783 | 0.8333 | 0.4762 | — | — | 4.06 | 3.37 |
| KD-only τ=10 | 0.4678 | 0.7051 | 0.8299 | 0.4643 | 0.4555 | 0.8130 | 2.76 | 2.33 |
| KD-only τ=20 | 0.4671 | 0.6751 | 0.7834 | 0.4368 | 0.4515 | 0.7569 | 1.94 | 1.83 |
| SUP + 0.3·KD τ=1（管线权重/温度） | 0.4151 | 0.6664 | 0.8110 | 0.4413 | 0.3995 | 0.8303 | 4.57 | 4.47 |
| SUP + 0.3·KD τ=5 | 0.4364 | 0.6866 | 0.8330 | 0.4542 | — | — | 4.23 | 4.17 |
| SUP + 1.0·KD τ=5 | 0.4464 | 0.6973 | 0.8364 | 0.4636 | — | — | 4.09 | 3.99 |
| SUP + 1.0·KD τ=10 | 0.4575 | 0.6902 | 0.8406 | 0.4707 | 0.4295 | 0.8449 | 3.77 | 3.65 |
| SUP，3 epoch | 0.4546 | 0.7085 | 0.8594 | 0.4759 | 0.4265 | 0.8695 | 6.21 | 5.85 |
| KD-only τ=10，3 epoch | 0.4558（峰值 0.4896 @ step 150） | 0.6964 | 0.8386 | 0.4837 | 0.4374 | 0.8122 | 3.35 | 2.49 |
| SUP + 1.0·KD τ=10，3 epoch | 0.4506 | 0.6976 | 0.8517 | 0.4706 | 0.4384 | 0.8595 | 4.76 | 4.51 |
| 对照：SUP，无随机负样本 | 0.3710 | 0.5965 | 0.7412 | 0.4365 | 0.3526 | 0.7634 | 4.65 | 4.43 |
| 对照：SUP，原学习率 1e-6/1e-5 | 0.2771 | 0.4537 | 0.6114 | 0.2771 | 0.2772 | 0.6093 | 0.34 | 0.24 |
| 对照：SUP，尺度 1（softmax 平坦） | 0.0518 | 0.1656 | 0.3406 | 0.1705 | 0.0509 | 0.3428 | 15.75 | 13.72 |
| 对照：复现 R10 配方 | 0.0033 | 0.0150 | 0.0334 | 0.1041 | 0.0026 | 0.0247 | 5.45 | 1.68 |

三种子（13/14/15）均值 ± std，1 epoch，见第 0 节表。

### 3.3 读法

- **上限问题的答案**：同一个双线性学生、同一份数据，SUP 就能从 0.27 到 0.43，KD 到 0.47–0.49。
  在 teacher 所重排的 C150 池内学生达到 0.46–0.48，与 teacher f0 的 0.4633 持平。结构上限至少在 0.49 以上，目前远未触及。
- **蒸馏策略的答案**：τ 是决定性的超参。τ=1 的 KD-only 甚至比 SUP 差（0.371 vs 0.418），τ=10 比 SUP 高 4–5pp，
  且三个种子一致。管线里的 `kd_weight=0.3, temperature=1` 这组配置与 SUP 没有可测差异。
- **KD-only 的代价**：KD 项只覆盖池内候选，没有随机负样本约束全局几何，所以 test R@150 比 SUP 低（0.81 vs 0.85）。
  `SUP + 1.0·KD τ=10` 两边都拿到（R@10 +2.8pp，R@150 最高）。
- **多 epoch**：3 个 epoch 下 KD 的 dev R@10 在 step 150 见顶后小幅回落，步间波动 ±2pp；正式跑需要学习率衰减或快照平均，
  以及按 dev R@10 选点（现有 `native_selection` 规则已经是 dev 驱动的）。
- **三个对照**分别证明：随机负样本值 +4.7pp（全库 R@150 +8pp）；学习率 ×100 是必要的；尺度 1 的平坦 softmax 本身就不稳定——
  损失只能靠放大分数来下降，R 的整个谱被吹大（σ₁=15.8），即使有随机负样本也崩。

## 4. 有希望的方案（按优先级）

### 方案 A（前提）：把学生训练配方修到「能学」

不改损失定义，只修四件事；探针已验证每一件都是必要的：

1. **logit 尺度**：学生分数乘固定尺度 s ≈ 20（或可学习的 log-scale 标量，初值 log 20）。C1 现有的 `10*sigmoid(raw)` 是同一思路但
   C2 用的是 `raw`。等价说法：rank_mass / KL 的学生端温度 = 1/20。
2. **随机负样本**：C2 每个 query 在池外追加 ≥ 256 个均匀抽样的合法 target（C1 本来就有 31 个 uniform，C2 没有）。
   这是秩一均值漂移的直接解药。
3. **学习率**：R 1e-3 / P 1e-4 起步（预注册的 100 倍），带余弦衰减；1–3 个 epoch；按 dev R@10 选快照。
   注意 `pipeline.py` 至今没有把 `student.P_lr/R_lr` 传给 `train_student_c2`，sweep 的 `fresh_lr_runner.py` 是用子进程注入的，正式修要接上。
4. **漂移监控**：anchor 的逐元素 MSE 对一百万个元素取平均后是 1e-6 量级，起不到任何约束；要么去掉，要么换成
   `σ₁(R−I)` 的谱约束。至少把 `σ₁/σ₂(R_QT−I)` 和全库 dev R@10 写进每个快照的日志，崩溃在 50% 快照就能看出来。
5. **向量化**：探针 198 步 8 秒；管线逐 query 循环 20 分钟。这不只是效率——它让多 epoch、多种子、τ 扫描变得可做。

预期：仅 SUP 就把直接 R@10 从 0.27 提到 ≈ 0.43，R@150 从 0.58 提到 ≈ 0.85。

### 方案 B：修蒸馏信号本身

在 A 的基础上：

1. **温度**：teacher 端 τ ∈ [5, 20]，使 teacher logit std / τ ≈ 学生 logit std（≈ 1）。探针最优 τ=10。
   更稳的写法是把 teacher logits 按列表做 z-score 标准化再乘学生尺度，这样不依赖 teacher 的绝对量级。
2. **权重**：KD 权重 ≥ 1（而非 0.3）。探针里 KD-only 的 R@10 最高，`SUP + 1.0·KD` 最均衡。
3. **保留 SUP 的随机负样本项**作为全局几何的锚（KD 项本身覆盖不到池外）。
4. 预期收益：相对 A 的 SUP 再 +3–5pp dev R@10（三种子一致）。

### 方案 C：扩大 KD 的覆盖面

KD 现在只在 ≈ 753 个池内候选上定义。两种扩展，成本都可控：

1. **为随机负样本缓存 teacher 分数**：每个训练 query 固定抽 256 个随机 target，用 teacher 一次性打分并写进
   `teacher_logits_cache`（现有缓存是 9.5M 对，新增 3.2M 对，一次离线推理）。之后 KD 项和 SUP 项覆盖同一个候选集，
   KD-only 也能约束全局几何，有望同时拿到 KD 的 R@10 和 SUP 的 R@150。
2. **Top-K 聚焦 KD**：teacher 的 softmax 熵只有 1.1 nat，几乎所有质量在前几十个候选；对 teacher top-50 做 KL、其余候选只做
   「低于 top-50」的 margin，梯度更集中、对 teacher 噪声更鲁棒。

### 方案 D：证据分支与论文叙事指标

探针只训了 direct 关系。论文叙事（`方案.md`：多模态证据补齐缺失属性、发现 join）落在 `Q_text / Q_image / text_T / image_T`
和路径聚合上，这些关系在 C2 里用的是同一套坏配方，`evidence_sup_loss ≈ 6.4` 同样平坦。应当：

1. 把 A+B 的尺度、随机负样本（对证据对象也抽）、温度同样施加到 `evidence_kd`（`aggregate_cqet` 后的列表 KL）。
2. 评估时单独报告 E 池覆盖（`E_gold`）、implicit-join query 的召回、以及 `CQET Real − Swap` 这类证据归因对照，
   不要只看 overall R@10。这部分没有探针数据，是待验证的假设。

### 方案 E：学生结构（只在 A+B+C 之后仍差 teacher 时再做）

目前的证据不支持先动结构。若 A+B 后学生在 C150 池内仍明显低于 teacher f0，可考虑：

1. 把 P 换成两层 MLP（仍是 dual encoder，ANN 兼容），能近似 teacher `global_relation` 里 `|g_q−g_t|` 这类非双线性项；
2. R 低秩 + 对角分解，减少 Adam 稠密更新带来的漂移面；
3. 查询侧/目标侧不同投影（P_q ≠ P_t），目前靠 R 承担非对称。

### 方案 F：管线层面的预期与验证方式

- 学生直接 R@150 从 0.58 → 0.84–0.87，意味着 C150 池覆盖（`C150_target_coverage`，现 0.62）会显著上升，teacher 重排的
  天花板（`oracle@10`）随之抬高。`primary_system`（teacher 在 C150 上的 R@10）应当上升，即使 teacher 本身不动。
  这一点未经管线验证，是推断。
- 验证顺序：先用探针脚本在 direct 上确认 A→B→C 的增量（每条臂 1 分钟），再改 `train.py` 的 C2（含证据分支），
  跑一次完整 9 阶段 + 冻结评估。KD gate 应当改为同时报告「学生自身直接召回」与「teacher 在学生池上的 R@10」。

## 5. 需要修正的记录

- `lrsweep/README.md`、`ADAPTATION.md` 的 ×10 结论和 `entitables_lr_sweep_bundle.tar.zst` 的 MANIFEST：
  「R10 与 LR0 召回逐位相同」「评估路径吸收扰动」→ 实为测量脚本读错文件；R10 两臂已崩（dev R@10 0.0025 / 0.0042）。
- `measure_k_curve.py`、`bootstrap_kd_gap.py`：rankings 路径须随 `--arm` 变化（R10 的学生从未进过冻结评估，需先跑评估再读）。
- `measure_c2_entitables.py`：`set(gt[query])` → `set(gt[query]["G"])`。
- 「KD 项生效但对排序无益」这句表述对 LR0 是对的（权重差 3e-4 相对量，确实可测但无意义），对 R10 不对。

## 6. 文件与复现

```
work/mmdd_stage1_v4_1_correctness_locked/lrsweep/probe_direct_kd/
  measure_direct_recall.py        六个 endpoint 的全库精确召回与打分尺度（表 2.1）
  collapse_trace.py               R10 各快照的 R@10 与 R−I 奇异值（表 2.3）
  probe_kd.py                     direct-only 重训探针（第 3 节）；--save 可导出学生
  teacher_on_new_pool.py          teacher f0 在旧池 / 新学生池上的重排与支路消融（第 7 节）
  students/                       两个探针学生（SUP+1.0·KD τ=10、KD-only τ=10，seed 13）
  results/endpoint_direct_recall.json
  results/collapse_trace.jsonl
  results/<arm>.jsonl             每 50 步一行；final 行含 test 指标
  results/teacher_newpool_*.json  第 7 节的重排对比（含 per_query）
  results/teacher_ablate_*.json   第 7 节的 token / global 消融
```

```bash
cd /tmp && conda run -n MMDD python .../probe_direct_kd/probe_kd.py \
  --name SUP_1KD_T10 --sup-weight 1 --kd-weight 1 --kd-tau 10 --seed 13   # ≈ 1 分钟（含加载）
conda run -n MMDD python .../probe_direct_kd/teacher_on_new_pool.py \
  --student .../students/sup_1kd_t10_s13.pt --split dev --out out.json       # ≈ 8 分钟
```

本机首次 CUDA 初始化偶发 `CUDA driver initialization failed`，重试即可。

## 7. 追问：学生已到 0.47，接下来该加强 teacher 还是继续训学生？（2026-10-01 补）

### 7.1 用的是哪个 teacher

V4.1 只有一条 teacher 链：`TA`（2 epoch，从头训）→ `TB_CQET`（1 epoch，从 TA epoch2 起）。TB 的训练列表是
`RawU ∪ RawDirect150 ∪ G ∪ U32`（`lists.py:464`），候选全部来自**原始 Qwen 嵌入**检索，没有学生参与；协议
`feedback_to_teacher = false`。KD 缓存 `teacher_logits_cache/identity.json` 指向 `TB_CQET/checkpoints/end.pt`，与评估
`rankings.TB_CQET.*` 的 `teacher_state_hash` 相同——第 3 节的探针、管线的 KD、以及「重排能到 0.46」用的是同一个 teacher。
`方案.md` 第 620 行附近描述的循环是「Frozen Teacher + Dynamic Candidate Distribution」：学生 ANN 挖难负例、**冻结** teacher 打分、再蒸学生；
teacher 本身不再训。更早的 `fresh_path` 里的 `refreshed_hard` 是 teacher 自己挖的。「学生挖难负例 → 再训 teacher」在现有代码里没有实现。

### 7.2 teacher 放到新学生的候选池上会怎样

`teacher_on_new_pool.py`：对每个 query 取新学生（探针，seed 13）的全库精确 top-150 作新池，旧池用 native_sup 的 C150；
teacher 走 `_teacher_c2_scores` 的 f0（不带证据路径）；fusion = 两边分数各自 z-score 后相加。

| split / 学生 | 旧池覆盖 | 新池覆盖 | 旧池：学生 / teacher / 融合 | 新池：学生 / teacher / 融合 |
|---|---|---|---|---|
| dev / SUP+1.0·KD τ=10 | 0.623 | **0.841** | 0.471 / 0.463 / 0.484 | 0.458 / **0.396** / 0.470 |
| dev / KD-only τ=10 | 0.623 | 0.830 | 0.464 / 0.463 / 0.481 | 0.468 / 0.384 / 0.475 |
| test / SUP+1.0·KD τ=10 | 0.625 | 0.845 | 0.442 / 0.456 / 0.456 | 0.430 / 0.386 / 0.456 |

旧池上的 teacher 0.4633 与管线 `SUMMARY.json` 的 `TB_CQET.f0 R@10` 逐位相同，打分路径复现无误。

三个事实：

1. 新学生把候选池覆盖抬了 22pp（0.62 → 0.84，这还只是 direct，没并证据路径）。
2. 旧池上 teacher 与学生打平；**新池上 teacher 比旧池掉 7pp**（0.463 → 0.396），尽管池里 gold 多了 22pp。新池里是新学生挑出来的
   困难负例，teacher 训练时只见过原始 Qwen 池的负例，分布错配后把大量难负例排到 gold 前面。implicit 与 explicit 两类 query 上都如此。
3. 融合在新池上比学生单独高 1.2–2.6pp：teacher 仍带有一点互补信息，但单独用它重排已经是负贡献。

### 7.3 teacher 的能力在哪条支路上

同一脚本 `--ablate`，dev，SUP+1.0·KD 学生的池：

| teacher 变体 | 旧池 R@10 | 新池 R@10 |
|---|---|---|
| 完整 | 0.4633 | 0.3957 |
| 置零 token transformer（只剩 `global_relation` MLP） | 0.4476（−1.6pp） | 0.4218（**+2.6pp**） |
| 置零 `global_relation`（只剩 token） | 0.1288 | 0.0495 |

与 v3.1 时的记录一致：teacher 的分数几乎全部来自作用在全局向量 `(g_q, g_t, g_q⊙g_t, |g_q−g_t|)` 上的 MLP，
token 级 cross-attention 在旧池只值 1.6pp，在新学生的难负例上是负贡献。也就是说当前的 teacher 实际上是
「同一个全局 Qwen 向量上的一个两层 MLP」，这解释了为什么一个双线性学生能追平它——它的「cross-encoder 优势」目前并不存在。

### 7.4 结论与建议顺序

- **继续蒸馏当前 teacher 的边际价值已经很小**：学生在池内与它打平，KD 剩下的作用是正则（KD 比 SUP 高 4pp 这件事仍然成立）。
- **但 teacher 现在是真正的瓶颈**，原因不是容量而是训练分布过时 + token 支路没学到东西。两级系统要有存在的理由，teacher 必须在
  新学生的池上明显强于学生，现在是反的。
- 建议顺序：
  1. 先把方案 A+B 落到管线（C1/C2，含证据分支）。这是前提，几分钟级成本，直接把池覆盖从 0.62 抬到 0.84+。
  2. 然后做「学生挖难负例 → 再训 teacher」：用新学生的 top-K 替换 TB 列表里的 `RawU ∪ RawDirect150`（或并集），重训 TB
     （现有实现 ≈ 2.4 h/epoch，向量化后应大幅缩短）。验收指标就是 7.2 那张表：teacher 在**新池**上的 R@10 要回到 ≥ 学生。
     过渡期可直接用 z-score 融合顶一下（+1–2.6pp，零成本）。
  3. teacher 在新池上重新领先之后，再蒸一轮学生（方案 B 的 τ、权重不变）。只有 teacher–student 出现差距，KD 才有东西可传。
  4. 面向论文的 teacher 加强：让 token 支路真正承担列级匹配（join key 重叠是 token 级信号，全局向量看不到），例如训练时对
     `global_relation` 加 dropout / 分阶段只训 token 支路、或改表格 tokenization 暴露列值重叠；同时把证据路径的
     `Real − f0`（现 +0.8pp dev、−0.6pp test）作为叙事核心指标单独盯。这一条是研究假设，没有探针数据。
- 没做的：新池只含 direct top-150，管线的 C150 还会并入证据路径；teacher 这里只用 f0，`Real` 视图会再多 ≈ 0.8pp。
