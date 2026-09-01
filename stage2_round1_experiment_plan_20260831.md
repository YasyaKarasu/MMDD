# Stage-2 第一轮实验计划：Oracle 正例可学习性与 Stage-1 条件对照（2026-08-31）

## 0. 给新 Goal 对话的执行指令

完整执行本文档中的“必须任务”，不要只给方案或命令。允许根据仓库实际接口做小幅调整，但不得改变实验口径。执行结束后必须提交：

- 可复现的代码和测试；
- 完整的数据审计与运行命令；
- `work/stage2_round1_20260831/RESULTS.md`；
- 机器可读的汇总指标和逐样本预测；
- 明确的继续/停止结论。

如果 Stage-1 的合法检索产物尚未准备好，先完成不依赖 Stage-1 的 Oracle 主实验，不要等待 Stage-1。检索条件对照可以标记为“等待合法产物”，但不得因此阻塞本轮主实验。

本轮是研究实验，不是生产化改造。实现应保持短小、显式、可审计，不增加分布式训练、自动重试或通用实验框架。

---

## 1. 背景与本轮要回答的问题

Stage 1 与 Stage 2 没有共享的可训练参数。当前 Stage 2 冻结本地 `Qwen3.5-9B`，只训练 `CandidateColumnScorer` 线性候选列头，因此从参数优化角度可以与 Stage 1 同期推进。

但现有 `src/train_stage2.py` 的训练样本依赖 Stage-1 retrieval：gold target 必须位于检索候选中，并且必须存在 `Q -> E -> T` evidence path。这样会把两个因素混在一起：

1. Stage 1 是否找到了正确 target/evidence；
2. Stage 2 在拿到正确 target/evidence 后，是否能学会选择正确列。

第一轮先把它们拆开，回答三个问题：

1. **可学习性**：直接使用数据集 gold target 和已经审核通过的 gold evidence，Stage-2 候选列头能否稳定学会正确列？
2. **泛化性**：在固定的 dev/test 划分上，提升是否不仅是训练集记忆？
3. **Stage-1 条件落差**：若有合法的 Stage-1 retrieval，Oracle 条件效果到真实检索条件会下降多少，瓶颈在 Stage 1 还是 Stage 2？

本轮不回答完整的 row filling、FOCUS 定位、值生成及最终 joinability verification 效果。这些属于第二轮端到端实验，不能用本轮的列选择准确率替代。

---

## 2. 已知数据与现有资产

### 2.1 数据集

使用以下两个 canonical dataset root，不重建数据集：

```text
output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_queryonly_autocheck_v4_jsonrepair_luna_consensus
output_wdc_webtable_2000_qwen35_unified_autocheck_v2_luna_consensus
```

当前审计得到的 `model_recoverable_join_column` 正例数如下。新 Goal 必须重新计算并写入数据审计，不能只复制本表：

| 数据集 | train | dev | test | 合计 |
| --- | ---: | ---: | ---: | ---: |
| EntiTables | 4,901 | 469 | 461 | 5,831 |
| WDC | 1,782 | 101 | 115 | 1,998 |
| 合计 | 6,683 | 570 | 576 | 7,829 |

目前每条 recoverable qrel 都至少有一条 `evidence_recoveries` 记录。证据分布存在明显湖间差异：EntiTables 有较多含 image 的 query，WDC 几乎是 text-only。因此 WDC 的 image 分层只做描述，不据此下显著性结论。

### 2.2 本地模型

```text
hf_models/Qwen3.5-9B
```

不得联网下载或替换模型。默认 `bf16`；只有硬件不支持时才改用 `fp16`/`fp32`，并在报告中记录。

### 2.3 Stage-1 正式结果与旧流水线

- R5 正式 Stage-1 结论位于 `work/stage1_optimization_r5_20260829/FINAL.md`。
- R5 使用分湖 Student，不是一个共享 checkpoint：
  - EntiTables：`work/stage1_optimization_r5_20260829/task1_kd_tau_ablation/entitables/tau_1.0/student_path.pt`
  - WDC：`work/stage1_optimization_r4_20260829/taskM_entitables_teacher/per_lake/wdc/student_kd0.3/student_path.pt`
- `work/stage1_stage2_wdc2k_entitables20k_v4_20260828/` 中存在旧 mixed pipeline、两份 train retrieval 和未完成的 Stage-2 步骤。

旧 retrieval **不得直接当作本轮正式输入**。使用前必须验证：checkpoint SHA-256、gate、corpus、index、retrieval 中的 `student_checkpoint_sha256` 完全一致。已知旧 mixed pipeline 的最终 gate 状态发生过变化，因此仅可作为代码/命令参考，不可伪造或改写指纹来通过验证。

---

## 3. 实验假设

### H1：Stage-2 候选列头在 Oracle 正例条件下可学习

使用 gold target 和 gold evidence 后，训练后的 test `column_accuracy@1` 应明显高于无训练基线，并且 train/dev/test 不应出现极端断层。

### H2：Stage-1 evidence recall 是端到端上限的重要瓶颈

若 Oracle 列准确率较高，但 Stage-1 条件下的 `joint_target_column_accuracy` 很低，则主要瓶颈是 Stage-1 没有把正确的 target/evidence 送到 Stage 2，而不是候选列头完全学不会。

### H3：两个湖不能只看混合平均

必须分别报告 EntiTables 和 WDC。混合训练允许共享一个候选列头，但 checkpoint 选择使用两湖 dev accuracy 的宏平均，避免 EntiTables 因样本更多而掩盖 WDC。

---

## 4. 固定实验口径

### 4.1 Oracle 正例定义

仅接受同时满足以下条件的 qrel：

- `reason == "model_recoverable_join_column"`；
- qrel 的 `split` 与当前加载 split 一致；
- 能找到对应的 query table、gold target table；
- 能找到同一 `query_table_id + target_table_id` 的已物化 `evidence_recoveries`；
- evidence asset 能在 `bridge_assets` 中解析，image 文件真实存在。

每个样本的 gold column 使用现有 `local_column_index(...)` 映射，不允许假设 `source_column_index` 就是 target 的局部列位置。

### 4.2 Oracle evidence 选择

每个 query 最多取 4 个去重 evidence asset，规则必须固定且与 split 无关：

1. 分别按 `asset_id` 排序 text 和 image evidence；
2. 两种模态都存在时，先各取一个；
3. 其余 evidence 按 `asset_id` 的稳定顺序补足至 4 个；
4. 记录被截断前后的 evidence 数量和最终模态组成。

这是一种显式的 Oracle evidence policy，不是 Stage-1 检索排序。报告和 checkpoint metadata 必须写明 `training_source=oracle_positive` 与该 policy，防止将其误认为正式端到端模型。

### 4.3 数据划分

- 只用 `train` 更新参数；
- 只用 `dev` 选择 epoch、学习率或停止条件；
- 配置冻结后只评一次 `test`；
- 不得将同一 query 的不同 split 混用，也不得用 test 调参；
- 审计 source table / chain 是否跨 split。如发现泄漏，立即停止训练，先修复或解释数据问题。

### 4.4 模型与优化

- backbone：本地 `Qwen3.5-9B`，全程冻结并处于 eval mode；
- 可训练参数：仅 `CandidateColumnScorer`；
- evidence top-k：4；
- epochs：3；
- learning rate：`1e-3`；
- weight decay：`1e-4`；
- seeds：`13, 17, 23`；
- 主 checkpoint：按两湖 dev `column_accuracy@1` 宏平均选择；并列时依次选择 dev NLL 更低、epoch 更早者。

不要在第一轮做大规模超参数搜索。若默认设置不收敛，只允许增加一个诊断配置，并在结果中明确标为 post-hoc。

---

## 5. 必须任务

## Task A：数据审计与 Oracle loader

### A1. 实现

在 `src/mmdd_stage2/` 中实现可复用的 Oracle column-example loader，复用现有 canonical artifact 读取和 `Stage2ObjectIndex`，不要在 CLI 中复制数据解析逻辑。

推荐扩展 `src/mmdd_stage2/training.py`，至少提供以下能力：

- 从一个或多个 dataset root 加载指定 `split`；
- 从 qrels + evidence recoveries 构建 gold `EvidenceBundle`；
- 每条样本保留 dataset/lake、query ID、target ID、gold source/local column、evidence IDs、模态组成；
- 检查 ID 冲突和缺失资产；
- 输出审计计数，而不是静默跳过坏样本。

可以扩展 `src/train_stage2.py`，加入明确互斥的训练源：

```text
retrieved        # 现有行为，仍强制 --stage1-gate 和 retrieval 指纹检查
oracle-positive  # 本轮实验，直接读取 qrels + evidence recoveries，不伪造 Stage-1 gate
```

必须保持现有 retrieved 模式的安全门禁和行为兼容。不得为了运行 Oracle 实验而删除、放宽或绕过 `validate_stage2_gate(...)`。

### A2. 数据审计产出

写入：

```text
work/stage2_round1_20260831/data_audit.json
work/stage2_round1_20260831/DATA_AUDIT.md
```

至少包含：

- 每湖、每 split 的 qrel 数和最终可用样本数；
- 缺 query/target/evidence/图片文件的数量和 ID 清单；
- 每样本 evidence 数量分布；
- text-only、image-only、text+image query 数量；
- target 候选列数分布；
- gold column position 分布；
- source table 与 chain 的跨 split 交集；
- 旧 retrieval/gate 的指纹审计结果，但不修改旧文件。

### A3. 测试

在 `tests/test_stage2_verifier.py` 或一个聚焦的新测试文件中覆盖：

- train/dev/test 过滤；
- qrel 与 recovery 的 query/target 匹配；
- evidence 去重、稳定排序和 top-4；
- text+image 时至少各保留一个；
- `local_column_index` 映射；
- 缺失 evidence/image 时给出明确错误或审计失败；
- Oracle 模式不需要 gate，但 retrieved 模式仍必须 gate；
- 不同 dataset root 的 ID 冲突检测。

测试使用 `tmp_path` 和小型合成 fixture，不加载 Qwen、不需要 GPU。

---

## Task B：冻结 Reader 状态缓存

当前训练每个 epoch 都重新运行冻结的 Qwen reader。第一轮有 6,683 个训练样本和 3 个 seed，直接重复 backbone forward 会浪费大量算力。

实现一个简单、显式的 reader-state cache：每个样本只运行一次冻结 Qwen，保存每个候选列的 opening/closing marker state，之后三个 seed 和三个 epoch 只训练线性头。

缓存记录至少包含：

- dataset、split、query ID、target ID；
- gold column position 与原始 column index；
- `open_states`、`close_states`；
- evidence IDs 与模态；
- target 候选列数；
- model path、hidden dimension、dtype、top-k 和 evidence policy metadata。

建议目录：

```text
work/stage2_round1_20260831/reader_cache/oracle_positive/
```

要求：

- cache 写入应先临时文件再原子替换；
- 已完成且 metadata 完全匹配的 shard 可以复用；
- metadata 不匹配时明确拒绝，不静默混用；
- 不需要实现并发写、网络缓存或通用 cache 框架；
- cache 是生成物，不提交大 tensor 文件；
- 必须验证 backbone 所有参数 `requires_grad=False`，缓存阶段使用 inference mode。

先做每湖 `32 train + 16 dev` 的 smoke cache，验证无 OOM、marker 数与 target 列数一致，再构建完整 cache。

---

## Task C：Oracle 候选列训练与评测（本轮主实验）

### C1. 基线

在相同 test 样本上报告：

1. `uniform_random_expectation`：逐样本 `1 / candidate_column_count` 的平均；
2. `majority_column_position`：只用 train 拟合最常见 gold position，再用于 dev/test；
3. `epoch_0_seeded_head`：三个 seed 下未训练线性头的平均和标准差；
4. `oracle_positive_trained_head`：本轮主模型。

不得使用 qrel 的 `column_name` 直接与候选 target header 匹配作为可部署 baseline，因为 qrel label 在实际推理时不可见。可以把这种匹配仅作为“标签泄漏审计”，但必须单列为 non-deployable diagnostic。

### C2. 训练

使用两湖合并的 train reader cache。每个 epoch 让两个湖具有相同的抽样质量或明确的 round-robin/lake-balanced 顺序，避免 4,901:1,782 的自然比例完全主导更新。记录每湖实际看到的样本数。

每个 seed 保存：

```text
work/stage2_round1_20260831/oracle_positive/seed_<seed>/
  candidate.pt
  history.json
  dev_predictions.jsonl
  test_predictions.jsonl
  metrics.json
```

checkpoint metadata 必须含：

- `training_source: oracle_positive`；
- dataset roots；
- evidence policy；
- Qwen model path；
- seed、epoch、optimizer 参数；
- reader cache fingerprint；
- dev selection metric。

### C3. 指标

主指标：

- `column_accuracy@1`。

辅助指标：

- column MRR；
- column NLL；
- accuracy@1 按湖；
- accuracy@1 按 evidence modality bucket；
- accuracy@1 按候选列数 bucket；
- train/dev/test gap；
- 三个 seed 的 mean/std；
- 主模型相对两个确定性基线的 paired bootstrap 95% CI（10,000 次，seed 13）。

逐样本预测必须至少包含：dataset、split、query ID、target ID、gold/predicted column index、是否正确、候选列数、evidence IDs/modality、各列 logits 或 probabilities。

### C4. Smoke gate

在完整缓存和训练前先跑小样本 smoke：

- 每湖 32 train、16 dev；
- seed 13；
- 1 epoch；
- loss 为有限值；
- 线性头参数发生更新；
- Qwen 参数不发生更新；
- 同 seed 重跑得到一致结果；
- checkpoint 能加载并完成评测。

smoke 失败时不要直接跑全量。

---

## Task D：Stage-1 检索条件对照（有合法产物时必须做）

该任务不得阻塞 Task A-C。

### D1. 合法输入条件

每个湖分别使用与自身 checkpoint 匹配的：

- dev-gated `student-path` selection manifest；
- Student checkpoint；
- corpus；
- HNSW index；
- train/dev/test retrieval JSONL；
- retrieval 内正确的 `student_checkpoint_sha256`。

由于 R5 是分湖 checkpoint，不要把两个湖强行塞进一个 gate。应分湖生成 retrieval，再在 Stage-2 loader 层合并样本。若正在执行的 Stage-1 新一轮实验尚未产生合法 gate，就在报告中列出缺失项并停止 Task D，不得伪造 `stage2_allowed` 或 checkpoint SHA。

retrieval 固定：

```text
max_targets=10
top_k_evidence=4
path_result_k>=10
evidence_path_k>=4
```

### D2. 对照模型

至少评估：

1. Oracle-trained head 直接用于 Stage-1 retrieval；
2. 若每湖 train eligible 样本不少于 100，再训练 retrieved-positive head，并在对应 dev/test retrieval 上评估。

retrieved-positive head 仍只在 gold target 的列上优化，必须在报告中说明它没有学习 target negatives。

### D3. 指标

每湖、每 split 报告：

- `gold_target_retrieved@10`；
- `gold_target_with_evidence_path@10`（Stage-2 eligibility）；
- `column_accuracy_given_gold_target_and_evidence`；
- `selected_target_accuracy@1`；
- `joint_target_column_accuracy@1`；
- `joint_target_column_accuracy_all_queries`，未 eligible query 按失败计；
- Oracle evidence 到 retrieved evidence 的条件 accuracy 落差。

不要只报告“成功进入 Stage 2 的样本”的准确率；eligibility 分母必须同时给出，否则会产生选择偏差。

---

## Task E：报告、结论与下一轮决策

最终写入：

```text
work/stage2_round1_20260831/RESULTS.md
work/stage2_round1_20260831/summary_metrics.json
work/stage2_round1_20260831/commands.sh
work/stage2_round1_20260831/logs/
```

`commands.sh` 只记录实际成功执行的命令，不保存环境变量、token 或其他秘密。不得读取、检查或修改 `.env.openai`。

`RESULTS.md` 必须回答：

1. Oracle 正例条件下候选列头是否可学习？
2. EntiTables 与 WDC 是否表现一致？
3. 是否存在明显过拟合或位置偏置？
4. image evidence 是否有可描述的信号？WDC 样本不足时明确写不足。
5. 若完成 Task D，端到端上限主要受 eligibility 还是列选择限制？
6. 是否值得进入第二轮 FOCUS/row filling/生成评测？

---

## 6. 裁决规则

使用以下规则形成结论，不因结果不好而临时改门槛：

### A. 候选列头可学习，进入第二轮

同时满足：

- 两湖 test accuracy 均比 `majority_column_position` 高至少 10 个百分点；
- 三 seed 平均提升方向一致；
- 宏平均 dev-test gap 不超过 5 个百分点；
- 没有数据泄漏或标签直接进入模型输入。

下一轮执行真实 retrieval 下的 FOCUS、row filling、值生成和 final verification。

### B. Oracle 可学习，但真实检索条件明显受限

Oracle 达到 A 的可学习标准，但：

- `gold_target_with_evidence_path@10` 很低，或
- `joint_target_column_accuracy_all_queries` 主要被 eligibility 拉低。

结论应为“Stage-2 列头可用，Stage-1 evidence recall/quality 是当前瓶颈”。继续优化 Stage 1，再做第二轮端到端；不要通过只汇报 eligible subset 掩盖问题。

### C. Oracle 条件仍学不会

任一湖相对 majority baseline 提升不足 3 个百分点，或出现严重 dev/test 崩塌：

- 暂停完整 Stage-2 流水线；
- 抽查 reader prompt、gold local column 映射、marker states 和位置偏置；
- 不启动昂贵的 FOCUS/生成实验。

3 至 10 个百分点属于不确定区间：保留结果，第二轮先做错误分析而不是直接扩训练。

---

## 7. 推荐执行顺序

```text
Task A 数据审计与 Oracle loader
  -> 单元测试
  -> Task B 小样本 reader cache smoke
  -> Task C 小样本训练 smoke
  -> 完整 train/dev reader cache
  -> 3 seeds 训练与 dev 选模
  -> 冻结配置后构建/评测 test cache
  -> Task D 合法 Stage-1 条件对照（若产物已具备）
  -> Task E RESULTS.md 与裁决
```

test cache 可以在配置冻结后再构建，减少误用 test 的机会。

---

## 8. GPU 与并行约束

逻辑上 Stage 1 和本轮 Oracle Stage 2 可以同时推进；物理上只有在不同 GPU 或显存足够时才并行。

执行前用 `nvidia-smi` 确认设备占用：

- 若有空闲 GPU，将 Stage 2 固定到该卡；
- 若只有一张 GPU 且 Stage 1 正在占用，不要同时加载 Qwen3-VL-Embedding-8B 和 Qwen3.5-9B 导致 OOM；改为错峰构建 reader cache；
- reader cache 完成后卸载 Qwen，线性头训练可以低成本运行；
- 报告实际 GPU、峰值显存、reader cache 耗时与线性训练耗时。

不得为了追求并行而降低数据正确性、跳过 gate 或覆盖现有 checkpoint。

---

## 9. 验证命令

实现后至少运行：

```bash
conda run -n MMDD python -m pytest tests/test_stage2_verifier.py -q
```

如果新建了聚焦测试文件，也运行对应文件。若改动影响 `mmdd_stage1.selection` 或 retrieved 模式，再运行相关 Stage-1 控制测试。

在新 CLI 完成后，以 `--help` 输出为准记录真实命令。预期形态可以类似：

```bash
conda run -n MMDD python src/train_stage2.py \
  --training-source oracle-positive \
  --dataset-root \
    output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_queryonly_autocheck_v4_jsonrepair_luna_consensus \
    output_wdc_webtable_2000_qwen35_unified_autocheck_v2_luna_consensus \
  --model-dir hf_models/Qwen3.5-9B \
  --device cuda:0 --dtype bf16 \
  --top-k-evidence 4 --epochs 3 \
  --learning-rate 1e-3 --weight-decay 1e-4 \
  --seed 13 \
  --output work/stage2_round1_20260831/oracle_positive/seed_13/candidate.pt
```

这只是目标接口示例。执行者必须先实现并测试所需的 Oracle split、cache、dev selection 和 evaluation 能力，再记录最终真实命令。

---

## 10. 明确不做的事项

- 不重训或修改 Stage 1；
- 不更改 R5/R8 现有结果与 checkpoint；
- 不伪造 Stage-1 gate、SHA 或 retrieval provenance；
- 不用 test 调参；
- 不解冻 Qwen3.5；
- 不在第一轮做 FOCUS 超参数搜索；
- 不把 Oracle gold target/evidence 指标描述为端到端效果；
- 不提交大型 reader cache、模型权重或逐层 hidden states；
- 不读取或操作 `.env.openai`。

本轮成功的定义不是“训练跑完”，而是得到一个数据口径清晰、可复现、能区分 Stage-1 eligibility 与 Stage-2 列选择能力的结论。
