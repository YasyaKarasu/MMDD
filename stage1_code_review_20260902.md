# Stage-1 代码审查报告：冗余设计 / 过度工程 / 设计冲突

日期：2026-09-02
审阅范围：`src/mmdd_stage1/` 全部模块（`data.py`、`construction.py`、`mining.py`、`models.py`、`objectives.py`、`training.py`、`retrieval.py`、`retrieval_aligned.py`、`evaluation.py`、`scoring.py`、`selection.py`、`teacher_logits.py`、`teacher_rerank.py`、`features.py`、`pca.py`、`artifacts.py`、`checkpoints.py`、`protocol.py`、`significance.py`、`workflow.py`、`evidence_diagnostics.py`）以及主要 CLI 入口（`train_stage1.py`、`evaluate_stage1_r3_baselines.py`、`cache_stage1_features.py`、`refresh_stage1_hard_negatives.py`）。

**原则**：这是科研代码，目标是让逻辑尽可能简洁。以下所有建议默认**不改变现有程序行为/数值输出**，只做去重、删死代码、合并重叠抽象、统一不一致的写法。凡是"落地会改变输出数值或错误行为"的条目，均已单独标注在第零节，需要用户先确认设计意图再决定是否修改。

所有"未使用/死代码"判断均已用 grep 在 `src/`、`tests/`、`scripts_old/` 全量交叉验证，避免误判。

---

## 零、需要用户先拍板的疑似 Bug / 设计冲突（不要直接改）

这些条目修复后会改变输出数值或程序的错误处理行为，请先确认哪一侧是"正确定义"，我们再决定怎么改。

| # | 位置 | 问题 |
|---|---|---|
| Z1 | `objectives.py:102-105` vs `retrieval.py:577-579` | 训练期与检索期的 `comb_mnz` 聚合公式不一致：训练侧乘的是**有效证据条数**（`evidence_mask.sum`），检索侧乘的是**非零分数条数**（`value != 0.0`）。当某条路径分恰为 0 时两者数值不同，即"训练目标"与"评测/检索打分"对同一配置给出不同结果。 |
| Z2 | `teacher_logits.py:44-50`（缓存文件名后缀）vs `teacher_logits.py:456-461`（加载校验） | 缓存文件名只在特定聚合方式下把 `temperature`/`power` 编入文件名，但加载时却对全部 4 个字段做严格比较。用同一 `aggregation` 换 `temperature` 会命中同一缓存路径却校验失败，抛出硬报错而非正常 cache miss。 |
| Z3 | `train_stage1.py:1249`（teacher-edge 阶段 `args.teacher_table_tokens_per_group or 1`）vs `train_stage1.py:1280`（teacher-path 阶段直接传参） | 同一个 `--teacher-table-tokens-per-group` flag 在两个 stage 下语义不同：teacher-edge 省略时会把已加载 checkpoint 的值强制覆盖成 1；teacher-path 省略时会沿用 checkpoint 保存值。且 CLI help（`:1636-1641`）描述的是后一种行为，与 teacher-edge 实际代码矛盾。 |
| Z4 | `train_stage1.py:1188-1191` | `--teacher-rerank` 模式下 `--primary-metric` 默认值 `"recall@10"` 必然导致第一个 epoch 抛 `KeyError`（该 stage 的 gate_metrics 只有 `dev_loss/teacher_rerank/raw_direct/spearman` 顶层键，没有裸的 `recall@10`）。正确用法需要显式传 `teacher_rerank.recall@10` 之类的作用域前缀，但代码没有在参数校验阶段拦截这个必崩配置。 |
| Z5 | `train_stage1.py:111-124`（`_per_dataset_gate_results`） | student 侧支持"扁平 key"和"嵌套 key"两种 metric 取值方式，raw 侧只支持嵌套 key。当 gate 的 metric 名不含 `"."`（扁平写法）时，raw 侧 per-query 值恒为 `None`，导致 bootstrap CI 静默跳过、悄悄退化成 `gate_mode="point_estimate_legacy"` 点估计，且没有任何告警。 |
| Z6 | `data.py:246-252` / `:381-387` | `teacher_logit_mode` 的回退链是"record → 专用 metadata key → 通用 `teacher_logit_mode`"三级，而 `teacher_ensemble_alpha` 的回退链只有两级，缺少通用 `"teacher_ensemble_alpha"` 这一级；但 `refresh_stage1_hard_negatives.py:262` 恰恰把值写进了这个通用键里，导致该值在当前代码路径下实际永远读不到。另外 `data.py:250` 的 `"teacher_edge_ensemble_alpha"` key 全仓库没有任何生产者写入，是死键。 |
| Z7 | `construction.py:492` | `for evidence_type in {asset_types[e] for e in positive_evidence_ids}:` 用 `set` 字面量迭代驱动带 `seed` 的随机选择，而这个文件其余地方都刻意维护可复现性（`sorted(..., key=...)`、按 query 播种的 `random.Random(f"{seed}:{query_id}")`）。set 迭代序依赖 `PYTHONHASHSEED`，同一个 seed 在不同进程下可能选出不同的 fallback evidence。 |
| Z8 | `retrieval.py:739-778`（`fuse_ranked_channels`） | 该函数**原地修改**传入的 `result` dict 的 `"score"` 字段。`evaluation.py` 对同一批 result 对象连续调用它三次（`fused`/`fused_e0`/`fused_e005`），导致这些 dict 的 `"score"` 字段最终全部被最后一次调用覆盖。当前下游只读 `target_id`/`paths`，指标不受影响，但这是隐蔽的跨调用耦合，任何未来读取 `result["fused"][i]["score"]` 的代码都会静默拿到错误值。 |
| Z9 | `construction.py:376` vs `construction.py:443-445` | 两处看似同构的 "取 evidence，缺失时回退到 `evidence_by_target[target_id]`" 表达式，`key` 的取值范围其实不同（前者可能命中 `recovery_evidence` 里的非 evidence-positive 条目），并非真正等价，只是写法相似容易被误合并。 |
| Z10 | `retrieval_aligned.py:160-161`（`align_target_record` 有配额溢出校验）vs `align_edge_record`（无对应校验） | `align_target_record` 在正例+人工负例超过 `list_width` 时会显式报错；`align_edge_record` 没有同样的校验，超额时 `_raw_negatives` 会静默返回空列表，最终可能产出长度超过 `list_width` 且不报错的候选列表。 |

---

## 一、跨模块的结构性问题（高优先级）

### 1.1 证据聚合公式（PathAggregator 语义）三处独立实现
- `objectives.py:53-107`（训练期张量版）、`retrieval.py:532-580`（检索期 Python 标量版 `_aggregate_path_channels`）分别实现了 logsumexp / max / topk_mean / topk_sum / softmax_weighted_mean / power_mean / comb_mnz 七种聚合方式。两者必须手动保持同步，目前已经出现分叉（见 Z1）。
- **建议**：先不改数值，仅在两处加交叉引用注释；长期可把 Python 标量版重写为张量版的参考实现，用测试锁定一致性。

### 1.2 `PathAggregator` 的 4 个字段（`evidence_aggregation`/`evidence_top_k`/`evidence_temperature`/`evidence_power`）被至少 6 处手写比较或重建
- `training.py:657-670`、`train_stage1.py:447-454`（`_validate_hard_provenance`）、`teacher_logits.py:35-41`（`_teacher_score_config`）、`data.py:320-329`（从 metadata 重建同一个 4 字段 dict，且默认值 `1.0`/`2.0` 与紧邻的 `data.py:338-340` 完全重复）、`evaluate_stage1_r3_baselines.py:604-607`（手写展开而非直接用 `aggregator.config()`）、`checkpoints.py:56-77`（`load_path_aggregation` 与 `load_path_aggregator` 功能重叠，前者已是死代码）。
- `data.py:342-353` 还重新实现了 `PathAggregator.__init__`（`objectives.py:32-39`）已有的校验，`checkpoints.py:60` 又做了第三份（只校验一半字段）。
- **建议**：
  - `TeacherScoreConfig(**aggregator.config())` 替代 `_teacher_score_config` 手工构造；
  - `data.py:320-329` 直接 `raw_score_config = metadata`（字段与默认值逐位相同，等价替换，删 6 行）；
  - `evaluate_stage1_r3_baselines.py:604-607` 换成 `**aggregator.config(),`；
  - `checkpoints.py` 删除死代码 `load_path_aggregation`（唯一调用方是它自己的测试，改测 `load_path_aggregator(...).evidence_aggregation/.top_k` 即可）；
  - `training.py:660-666` 的逐字段比较改为 `TeacherScoreConfig` 的 dataclass `__eq__`（`teacher_logits.py:373` 已经这么用了）。

### 1.3 `checkpoint_fingerprint` / `write_json` 存在两条 import 路径
- 真实定义在 `mmdd_stage1/artifacts.py`；`retrieval.py:16` 转导出 `checkpoint_fingerprint`（自身从未使用）、`selection.py:13` 转导出 `write_json`（自身从未使用）。结果全仓库约 25 个脚本从 `retrieval` 导入指纹函数、约 15 个脚本从 `selection` 导入 `write_json`，另一批（`train_stage1.py`、`workflow.py`）直接从 `artifacts` 导入——同一个函数两种写法并存。
- **建议**：把所有 `from mmdd_stage1.retrieval import checkpoint_fingerprint` / `from mmdd_stage1.selection import write_json` 统一改成 `from mmdd_stage1.artifacts import ...`，删除两处转导出。纯 import 路径调整，零行为变化（改动面较大，建议放在收尾阶段一次性做）。

### 1.4 checkpoint 的"写"与"读"分居两个模块，且相关能力散落三处
- `training.py:1221-1236` 的 `checkpoint()` 序列化逻辑与 `checkpoints.py` 的 `load_*` 系列分处两个文件，`format_version=1` 这个契约的两端因此分离。
- `selection.py` 又额外持有 `checkpoint_artifact_paths` / `CheckpointManager`（保存、拷贝、剪枝 checkpoint 文件），与 `checkpoints.py`（读取/反序列化）职责重叠但未合并。
- **建议**：把 `training.py` 的 `checkpoint()` 移到 `checkpoints.py`（纯搬移）；`selection.py` 的 `checkpoint_artifact_paths` + `CheckpointManager` 也建议迁到 `checkpoints.py`，`selection.py` 只保留 gating 相关逻辑（`metric_value`/`GateDecision`/`MetricGate`/`load_stage1_selection`/`validate_stage2_gate`）。需同步改 `train_stage1.py:44`、`run_stage1_rounds.py:189`、`tests/test_stage1_training_control.py:33` 的 import 来源。

### 1.5 JSONL 记录 schema（target/edge record）有 3 个独立写入者、1 个独立解析者
- 写：`construction.py:456-467`（edge 字面量）、`construction.py:551-561`（target 字面量）、`mining.py:250-285`（`_target_record`/`_edge_record`）、`retrieval_aligned.py:222-231`（又一份 target 字面量）。
- 读：`data.py:207-393`。
- 已发生实质不一致：`construction.py` 无条件写 `"split"` 字段，`mining.py` 仅在非 None 时才写。
- **建议**：把 `mining.py` 的 `_target_record`/`_edge_record` 上移到 `data.py`，改为公开的 `target_record(example)`/`edge_record(example)`，与读取函数放在一起（读写成对，便于对齐字段）；`mining.py` 改为导入使用；`construction.py`/`retrieval_aligned.py` 至少共用同一份字段名常量。

### 1.6 一批 CLI 通用逻辑在多个脚本里各写一份
| 逻辑 | 重复位置 |
|---|---|
| `--evidence-modality-weights` 解析 | `train_stage1.py:160-172`、`evaluate_stage1_r3_baselines.py:57-67`、`retrieve_stage1.py:21`、`evaluate_stage1_teacher_retrieval.py:28`（4 份逐字相同） |
| `--recall-ks` 解析 | `train_stage1.py:175-182`、`evaluate_stage1_r3_baselines.py:40-44`、`evaluate_stage1_selection.py:27`、`sweep_stage1_fusion.py:27`（4 份不同实现，且库里 `evaluation.py:21-26` 的 `_validate_recall_ks` 本可直接复用） |
| `PathAggregator` "CLI 覆盖或沿用 checkpoint 保存值" | `train_stage1.py:997-1020`、`evaluate_stage1_r3_baselines.py:441-461`（20+ 行 × 2） |
| `device` 解析三行式 | 13 个脚本重复，如 `train_stage1.py:935-939`、`evaluate_stage1_r3_baselines.py:373-375`、`cache_stage1_features.py:698` |
| dev 样本加载 | `train_stage1.py:328-337`、`evaluate_stage1_r3_baselines.py:70-78`、`evaluate_stage1_selection.py:34`、`run_stage1_r3_sweeps.py:27`、`run_stage1_r6_sweeps.py:131`（5 份） |
| ensemble alpha 区间校验 `[0,1]` | `teacher_rerank.py:43-44/212-213/277-278`、`teacher_logits.py:336-338`（4 份） |
| `"ensemble" if alpha is not None else "teacher"` | `teacher_logits.py:411/548-550/578/589`、`refresh_stage1_hard_negatives.py:268`（5 份） |

- **建议**：在 `mmdd_stage1` 里新增（或复用既有）工具模块，把上述逻辑各收敛为一个公开函数，CLI 脚本统一 import。函数体基本可以照搬现有实现之一，逐行零行为变化。

### 1.7 检索参数从入口到底层被透传 4~5 层，同一组 ~20 个参数抄了 5 遍
- `retrieval.py` 内部：`retrieve_zero_one_hop`（`:805-833`）→ `retrieve_zero_one_hop_detailed`（`:1188-1212`，本身只是 `..._detailed_many([qid], ...)[0]` 的 52 行参数表包装）→ `retrieve_zero_one_hop_detailed_many`（`:1005-1030`，递归分批时 `:1070-1100` 又把全部 kwargs 抄一遍）。
- `evaluation.py:146-174 + 207-231` 的 `evaluate_student_retrieval` 再抄一遍同一参数表。
- 默认值同时出现分叉：`retrieval.py` 侧 `fusion_mode="rrf", evidence_weight=1.0`，`evaluation.py` 侧 `fusion_mode="weighted_rrf", evidence_weight=0.05`（可能是有意为之，但无注释说明"权威默认"是谁）。
- **建议（保守）**：删除 `retrieve_zero_one_hop_detailed`，两个调用点直接改调 `..._detailed_many([qid], ...)[0]`；把 `..._detailed_many` 内部的校验与参数推导留在外层，主体逻辑抽成内部函数 `_retrieve_batch(...)`，递归分批改为普通循环，避免 kwargs 重复抄写与重复校验。
- **建议（更彻底，需评估改动面）**：引入 `FusionConfig` frozen dataclass 承载这组参数，各层只传一个对象。

---

## 二、`models.py` / `objectives.py` / `training.py`

### 2.1 `models.py`
- **`"orthogonal"` 与 `"random_orthogonal"` 是完全相同的代码路径**（`models.py:16-23, 467, 508-513, 538-543`），全仓库只有 `random_orthogonal` 被实际使用。建议合并判定条件为一个模块常量集合，或直接从 `STUDENT_INITIALIZATIONS` 移除从未使用的取值。
- **`compress()` 是 `compress_many()` 的单对象重复实现**（`models.py:205-227` vs `255-324`），table 分支的 `token_kinds` 构造逻辑写了两遍，且 `compress()` 在生产代码中无调用点（仅测试使用）。建议把 `compress()` 改为对 `compress_many()` 的薄委托。
- **relation 打分公式被写了 4 遍**：`score_embeddings`（`:624-641`，仅测试使用）、`score_embedding_matrix`（`:643-661`）、`score_pairs` 内联块（`:711-720`，与 `score_embeddings` 逐字相同）、`relation_query`/`index_vector`（`:725-751`）。建议抽 `_score_projected` / `_score_projected_matrix` 两个私有方法供其余四处复用。
- **`IdentityStudentJoinabilityModel` 与 `ProjectedIdentityStudentJoinabilityModel` 逐字重复**（`:754-855`）的 `relation_query`/`index_vector` 逻辑，可抽 mixin 共享（`project()` 的差异化行为保留在各自类中）。
- **`StudentANNModel` 联合类型协议不完整**：`retrieval.py` 中 3 处用 `getattr(model, "ann_dim"/"relation_param", 默认值)` 兜底，本质是接口没定义好。建议给两个 Identity 类补上 `relation_param` 类属性与 `ann_dim` property。
- relation 初始化噪声硬编码 `0.01`（`:544, 550`），与可配置的 `initialization_noise_std`（默认同为 0.01，但只作用于投影矩阵）语义脱节；`freeze_projections` 被赋值两次（`:492` 与 `:528`）；`_compression_buckets` 的 `max_padded_tokens` 形参从未被覆盖（`:229-232`），可提为常量并去掉形参；`score_pairs`/`compress_many` 对 `object_type` 的规范化方式不一致（前者用原始字符串作 key，后者用 `normalize_object_type`）。

### 2.2 `objectives.py`
- **`PathAggregator` 继承 `nn.Module` 但没有任何参数/缓冲**，全仓库无 `.to()/.state_dict()/.parameters()` 调用，纯粹是为了 `aggregator(...)` 这种调用语法。建议改为 `@dataclass(frozen=True)` + `__call__`，同时天然获得值相等语义（可直接服务 1.2 节的比较统一）。
- `positive_indices` 与 `positive_mask` 是同一信息的两套表示：`optional_listwise_cross_entropy` 内部会对同一批数据解析两次 mask（先 `_usable_list_rows` 后 `listwise_cross_entropy` 各自调用 `_resolve_positive_mask`），且 `positive_mask` 非 None 时 `positive_indices` 被完全忽略却仍强制传参。建议提取共享的按 mask 计算的内核函数消除二次解析。
- `forward` 里 `comb_mnz` 分支靠隐式 `else` 兜底（`:101-105`），新增聚合方式忘记加分支时会被静默当作 comb_mnz；建议改成显式 `elif` + 最终 `else: raise AssertionError`。
- `distillation_kl` 对 teacher 分布 softmax 与 log_softmax 各算一次（`:180-182`），可用 `log_softmax` 后 `.exp()` 省一次计算（数值上有 ~1e-7 级差异，追求逐位复现可跳过此项）。

### 2.3 `training.py`（问题最集中的文件）
- **四个 `train_*` 函数（`train_teacher_edges`/`train_teacher_paths`/`train_student_edges`/`train_student_paths`）共享同一套 epoch 脚手架**（`:737-1218`），包括 rng 播种、`sample_mixed_epoch`、进度条刷新、`epoch_bar.set_postfix` 三元展开（四处逐字相同）、`_epoch_record`/`_finish_epoch`。建议提取 `_run_training_loop(model, ..., step_fn, dev_fn)` 通用骨架，四个函数退化为构造 `step_fn`/`dev_fn` 的薄壳。这是收益最大但改动面也最大的一项，建议放在最后单独排期验证。
- **每个指标一对 `pending_/values` 列表的样板代码重复 9 次**（`train_student_paths` 内部 `:1055-1194`，`train_student_edges`/`train_teacher_paths` 同构重复），本质是把已有的 `dict[str, Tensor]` 人工拆成多组同构变量。建议改为 `dict[str, list[...]]` 累加器统一处理，约省 70 行。
- **dev 评估函数与训练循环体逐字重复**约 25~40 行：`_student_edge_objective`（`:620-652`）与 `train_student_edges` 内联块（`:932-964`）；`_student_path_objective`（`:695-732`）与 `train_student_paths` 内联块（`:1088-1133`）。建议抽出共享的 `_edge_batch_objective`/`_path_batch_objective`，训练与 dev 各自套壳调用。
- `_student_edge_objective`/`_student_path_objective` 用 10~13 个位置参数传参且调用点全部按位置传递，与同文件其余函数的关键字风格冲突，建议加 `*` 强制关键字。
- **确认死代码**：`sample_balanced_epoch`（`:42-64`）在生产路径无调用（四个 `train_*` 均走 `sample_mixed_epoch`），仅测试使用；`_list_scores` 的 `positive_mask` 形参从未被传参使用（`:166-175`），且 `_target_teacher_scores` 明明需要这个能力却选择重新构造一个 `ListScores`（`:255-260`），说明该形参是遗留物。
- `_target_teacher_scores` 每个 batch 都白算两次 `target_positive_mask`，但下游只读 teacher 侧的 `.logits`，mask 从未被消费（`:243-268`），建议直接删除这两次计算。
- `_list_scores` 内联重复实现了 `scoring._mask`（`:172-174`），建议直接复用。
- `relation_param` 分支判定在 `training._student_relation_keys`（`:422-425`）与 `models.relation_parameters`（`models.py:579-587`）各写一遍，建议在 `StudentJoinabilityModel` 上加 `relation_keys()` 方法收敛。
- `student_relation_drift`/`student_projection_drift`（`:390-419`）是纯模型自省逻辑却放在 `training.py`，建议搬到 `models.py`。
- `_anchor_losses` 短路（`:438-440`，`anchor_weight==0 and evidence_weight==0` 时返回全零）会让 history 里的 `anchor_loss` 诊断字段在 CLI 默认配置（`--anchor-weight 0.0`）下恒为 0，虽然真实几何漂移并非 0——建议至少加注释说明这是"加权前观测量在默认配置下不可用"，不建议改变短路行为本身。
- 训练函数默认值不统一：`train_student_edges.distillation_weight` 有默认值 `1.0` 而 `train_student_paths` 同参数无默认值；`dataset_sampling_alpha`/`hard_fraction` 等参数的默认值在生产中永远被 `train_stage1.py` 的 `common` 字典覆盖、从不生效，容易让读者误判。建议四处默认值统一或去除。

---

## 三、`retrieval.py` / `retrieval_aligned.py` / `evaluation.py` / `scoring.py` / `selection.py`

### 3.1 死代码 / 纯转发（可直接删除）
- `retrieval.py:583-588` `_channel_ranks()` 全仓库无调用点，逻辑已被 `rank_detailed_paths`/`fuse_ranked_channels` 内联实现，直接删除。
- `teacher_rerank.py:19` `RECALL_KS = DEFAULT_RECALL_KS` 纯别名（同类别名还出现在 `run_stage1_r6_sweeps.py:38`、`run_stage1_r7_task_q.py:35`），建议直接用 `DEFAULT_RECALL_KS`。
- `retrieval_aligned.py:11-14` 的 `RawSearch` Protocol 全仓库无实际使用，项目其余地方表达同样约束用的是具体联合类型 `StudentANNIndices | RawEmbeddingANNIndices`，建议删除该 Protocol 统一写法。
- `evaluation.py:197-206` 中一个不做实质工作的 progress 条（纯 O(n) 列表推导包了进度显示），删除即可。

### 3.2 重复实现同一逻辑
- z-score 计算在 `retrieval.py:607-613` 与 `:917-938` 各写一遍（含相同的 `1e-12` 退化阈值），建议抽 `_zscore_stats`。
- `StudentANNIndices`（`:256-321`）与 `RawEmbeddingANNIndices`（`:399-449`）的 HNSW 索引加载、`search`/`search_many` 逻辑重复约 40 行，建议抽共享的模块级私有函数。
- `scoring.py:100-102` 的 `_mask()` 与 `training.py:166-175` 的 candidate mask 构造逐字重复；`scoring.py:560-571` 与 `training.py:227-238` 的 evidence 存在性 mask 构造也重复；建议 `training.py` 直接复用 `scoring` 中的对应函数。
- `scoring.py:176-181` 与 `:420-425` 的 hidden_dtype 推导表达式重复；`scoring.py:49-66` 的 `_device_features` 与 `teacher_rerank.py:138-149` 同名函数是同一逻辑的无缓存复制品，建议后者直接复用前者（传空 dict 作缓存）。
- `evaluation.py:65-84` 的 `_channel_metrics` 与 `teacher_rerank.py:86-136` 的 `_retrieval_metrics`/`_metrics_by_dataset` 是同一套"聚合 + 按 dataset 切分"的两份实现，且求均值写法不统一（`sum/len` vs `statistics.fmean`）。

### 3.3 冗余计算（主要在 `evaluation.py`）
- `fused` 通道指标被完整计算两遍（`:122` 与 `:125`），后者只是为了取两个 coverage 字段，建议只算一次复用。
- `_channel_metrics` 对每个 k 都重新计算一次 MRR 却只取 recall 部分（`:65-84`），5 个 k 就是 5 次浪费；`positive_evidence_hits` 对所有 `requested_k` 都计算但只有 `k=10` 被读取（`:256-267`），应加 `if requested_k == coverage_k:` 守卫。
- `retrieval_budget` 报告中的 `direct_k` 推导公式（`evaluation.py:292-294`，用普通乘法）与 `retrieval.py:1040-1046` 实际使用的 `math.ceil(gamma * k)` 是同一公式的第二份拷贝，当 `gamma` 为浮点数时会产生不一致（目前 `gamma` 恒为 int，暂未触发）。

### 3.4 设计不一致（建议统一写法，不改行为）
- `retrieval.py:361` 对 `self.relation_param` 做 `getattr` 防御，而该属性在 `:299` 已无条件赋值，属无意义防御，可直接改为属性访问。
- `retrieval.py:128-136` 的三元分支在所有可达路径下恒等价（`index_specs` 保证了 `source_type` 的取值与分支条件已经绑定），可塌缩成一行。
- 字符串枚举校验（`score normalization`/`path_edge_normalization`）散落 3~4 处（`retrieval.py:629, 663-666, 893-894, 1052-1057`），而 `fusion_mode` 已有正确范式（`FUSION_MODES` 常量 + `_validate_fusion_mode`），建议其余枚举也照此范式收敛。
- `retrieval_aligned.py:68-171` 的 `align_target_record`/`align_edge_record` 共享同一套"配额填充"骨架但独立实现，建议抽共享的 `_fill_from_raw(...)` 辅助函数（注意 Z10 提到的校验不一致需先确认）。
- `selection.py:81-90`（`checkpoint_artifact_paths`）在同一函数里混用了手工 stem/suffix 运算和 `with_suffix` 追加两种路径拼接习惯，建议统一。
- `selection.py:159-165` 同时容忍 JSON 数组/单对象/JSONL 三种格式，但生产侧只有单一格式的生产者（`retrieve_stage1.py`），属于没有实际收益的过度容错，且用 `except JSONDecodeError` 做控制流会把真正的文件损坏误判为"按 JSONL 重试"。

---

## 四、`teacher_logits.py` / `teacher_rerank.py` / `features.py` / `pca.py`

### 4.1 `teacher_logits.py`
- `_raw_edge_scores`（`:53-86`）构造的 `candidate_mask`/`positive_indices` 从未被下游消费（`_ensemble_list_scores` 只读 `.logits`），属死计算；`_ensemble_evidence_scores` 同理（`:235-257`）。建议简化为只返回 logits 张量。
- `score_and_cache_cosine_logits`（`:654-659`）借道 ensemble API（把同一份 raw 分数同时当 raw 和 teacher 传入、`alpha=0`）表达"纯 z-score"，可读性差且会白算一次 teacher 侧 z-score，建议抽 `_normalized_list_scores`/`_normalized_evidence_scores` 直接表达该语义。
- `score_and_cache_cosine_logits` 与 `score_and_cache_teacher_logits` 尾部的写回逻辑（`:562-617` vs `:660-711`）大段重复，建议抽 `_write_cache`/`_target_payload` 辅助函数。
- `tuple(float(v) for v in X[row, :count].cpu().tolist())` 模式出现 8 次，建议抽 `_row_tuple` 辅助函数。
- `has_teacher_logits`（`:508-514`、`:634-642`）被逐样本调用，每次都重新校验 ensemble alpha 与聚合配置，建议把校验提到循环外、内部改成谓词函数。
- `teacher_logit_mode` 是完全可从 `ensemble_alpha` 派生的冗余字段，却同时存在于 example 字段、缓存 payload、缓存文件名三处，各自维护一致性检查（`:341-349`、`:409-413`）——此项改动会影响已落盘的 JSONL/缓存 schema，需先确认无外部消费者后再考虑精简。

### 4.2 `teacher_rerank.py`
- `_teacher_scores`（`:154-193`）用"key 类型三元分支"处理"是否传入 score_cache"两种模式，建议统一成局部 dict 一种写法（`cache = score_cache or {}`），代码可减半，且能顺带修复 `missing_ids` 未去重导致的重复打分问题。
- "按分数排序取 ranking" 的 lambda 表达式重复 3 处（`:242-246, 307-314, 318-324`），"overall + by_dataset" 的三段式代码重复 3 处（`:329-352`），均可各抽一个小函数。
- `delta_k` 的动态 key 选择（`:354`）是无效的通用化：所有下游消费方都硬编码读 `"recall@10_delta"`，一旦 `recall_ks` 不含 10 就会 KeyError。建议固定该 key，`recall_ks` 不含 10 时不写入。
- `_retrieval_metrics`（`teacher_rerank.py:86`）与 `evaluation.py:84` 同名不同义，容易读错；建议改名为 `_rerank_metrics` 消歧。

### 4.3 `features.py`
- `estimated_feature_bytes`（`:318-340`）的两个分支单位口径不一致：eager 分支返回张量实际字节，index 分支返回磁盘文件大小，二者被当同一单位用于 `configure_hot_cache` 的预算计算，而 `cache_info()` 又用第三种口径（`_feature_bytes`）汇报实测值。建议至少在 docstring 中说明口径差异。
- `preload_embedding_matrix` 空 id 列表的早退分支漏了 `self._cache_bytes = 0` 归零（`:446-449`），非空分支有做（`:469-470`），会导致 `cache_info()` 报告残留旧值，属于可安全修复的小 bug。
- `from_path` 的 eager 分支丢弃了 `cache_size` 参数（`:236`），与 directory 分支不一致（当前无实际影响，因为 eager 存储不走 `_cache`），建议补齐参数传递以避免未来演进为真 bug。
- `configure_hot_cache` 的返回值与 `cache_info()` 用两组不同键名报告同一组指标（`planned_objects/estimated_bytes/access_coverage` vs `planned_hot_objects/planned_hot_estimated_bytes/planned_hot_access_coverage`），`train_stage1.py` 里两份都被落盘，建议二选一。

### 4.4 `pca.py`
- `compute_pca_projection`（随机化算法）与 `compute_pca_spectrum`（精确算法）是两套 PCA 实现，产出两种 artifact，`load_pca_projection` 被迫用 `is_spectrum` 分支兼容两者；且两者的 `explained_variance_ratio` 字段同名不同义（标量 vs 累计比例向量）。建议至少给 spectrum 侧字段改名消除歧义；是否能合并两条实现路径需先确认精度/性能要求。
- **隐性耦合**：`PCA_FORMAT_VERSION` 与 `PCA_SPECTRUM_FORMAT_VERSION` 两个独立版本常量，`probe_stage1_pca_dimensions.py` 用后者写入、`load_pca_projection` 却用前者校验同一个字段，目前两者数值恰好相同（都是 1）所以能跑通，但任一方 bump 会静默导致 spectrum artifact 无法加载。建议改用匹配的常量比较。
- `mean` 字段被两个函数计算、持久化，但全仓库没有任何地方读取它（Student 初始化不做中心化），属于死输出；且计算时 GPU/CPU 归属不一致（其余返回值都调用了 `.cpu()`，唯独 `mean` 没有）。此项改动会影响 artifact schema，建议先确认无外部消费者。
- 两个函数的输入校验（形状、有限性、非零方差）与清理逻辑（`del` + `torch.cuda.empty_cache()`）逐字重复 4 处，建议抽 `_validate_embeddings`/`_release_cuda`。

---

## 五、小型工具模块（`artifacts.py` / `checkpoints.py` / `protocol.py` / `significance.py` / `workflow.py`）与两个 CLI 脚本

- **`checkpoints.py` 的 `load_path_aggregation` 是死代码**（`:56-62`），只被自己的测试引用，生产代码全部使用功能更完整的 `load_path_aggregator`。建议删除，测试改为断言后者的属性。
- **`protocol.py` 里 `validate_r6_readonly_invariants`（`:23-43`）是 r6 轮次专用的一次性校验逻辑**，却放在通用的"协议"模块里，实际只被 `summarize_stage1_r6.py`（一份历史归档脚本）及其测试使用，与同模块中真正通用的 `validate_protocol_split`（被当前所有主要入口调用）性质不同。建议后续新增归档脚本时不要再往 `protocol.py` 里加轮次专属规则，这类校验更适合放在对应脚本自己的文件里。
- **`significance.py` 的 `paired_bootstrap_delta` 返回值里存在成对的重复键**（`"mean"`/`"delta_mean"` 同值，`"ci_low"`/`"ci95_low"`、`"ci_high"`/`"ci95_high"` 同值）。这是历史上不同批次脚本各自选用了不同键名造成的兼容层，目前被数十个 `summarize_stage1_r*.py` 一次性分析脚本按不同别名读取，改动面很大、收益有限，建议只在**未来新增代码**中固定使用一套命名（推荐 `delta_mean`/`ci_low`/`ci_high`），不建议现在动这个函数本身。
- `refresh_stage1_hard_negatives.py` 中"pending 校验用的 expected 字典"（`:142-159`）与"最终写入 metadata 的字典"（`:240-263`）共享 12 个同名键，两处分别手写，属轻度重复，可考虑抽取公共的"核心挖矿元数据"构造函数，但因为两处的字段差集不完全相同（后者多了 teacher 相关字段），收益有限，优先级较低。
- `cache_stage1_features.py` 中大量 `getattr(args, "x", default)` 写法初看像多余的防御，但经确认 `run_stage1_rounds.py:467` 会用手工构造的 `argparse.Namespace`（缺少多数可选字段）调用 `cache_stage1_features.run(...)`，因此这些 `getattr` 默认值是真实必要的，**不是**冗余设计，本次审查未发现需要改动之处。

---

## 六、`train_stage1.py` / `evaluate_stage1_r3_baselines.py` 内部冗余

### 6.1 `train_stage1.py`
- `_referenced_object_ids`（`:340-350`）与 `_object_ids`（`:353-361`）是同一函数的两个副本，前者可直接写成对后者的聚合调用。
- `preload_embeddings` 代码块在 `if`/`elif` 两个分支里逐字重复（`:1169-1179`），可下沉到分支结束后统一执行一次。
- `_EpochController.__call__` 里两次 `evaluate_student_retrieval` 调用（raw baseline 与 student，`:611-665`）约 22 个关键字参数中 20 个逐字相同，建议抽私有方法 `self._evaluate(indices, **extra)`。
- `record["gate"]` 字典在两条路径下分别构造（`:723-730` 与 `:742-749`），共享 6 个键，建议抽小函数统一。
- `MetricGate` 的内部状态（`bad_epochs`/`best_epoch`/`best_value`）被 `_EpochController` 直接改写（`:717-722`、`:770-772`），与 `MetricGate.observe`（`selection.py`）内部已有的同类逻辑并存，属于"两处定义同一份早停语义"，建议在 `MetricGate` 上补 `observe_ineligible()`/`force_best()` 方法收口。
- `_apply_argument_defaults` 被调用 3 次且每次重建整个 ~90 参数的 `ArgumentParser`（`:297, 479, 825`），经确认编程式调用路径（`run_stage1_rounds.py`）下 `run()` 里的那次已经覆盖了前两次的效果，前两次是 no-op，建议删除并给 parser 构造加缓存。
- `_add_feature_accesses` 的 `multiplier` 形参从未被传参使用（`:388-393`），建议删除。
- 三份输出清单（`history_payload`/`selection`/`summary`，`:1433-1550`）之间有大量同名同值的重叠键（`history_payload`与`selection`共享12个、`selection`与`summary`共享9个），建议先构造一个共享字典再用 `{**shared, ...}` 展开，输出 JSON 内容逐字节不变。
- `history_payload` 里 `kd_target_teacher_alpha` 与 `teacher_ensemble_alpha` 两个键写入同一个值（`:1452-1453`），后者经确认无任何读取方，建议删除该行。
- **确认死路径**：`--train-data` legacy flag 及其冲突检查（`:1561, 787-791`）全仓库无任何非 None 调用，`--teacher-ensemble-alpha` 弃用别名及其一致性检查（`:1767-1773, 826-834`）同样全仓库无调用，建议连同相关分支一并删除（后者需同步调整一处测试断言）。

### 6.2 `evaluate_stage1_r3_baselines.py`
- `run()` 里 13 处 `getattr(args, key, default)` 经确认全部是死防御——`args` 只可能来自本文件的 `parse_args()`（无任何编程式调用方），且每个 `getattr` 默认值都与对应 `add_argument` 的 `default` 完全一致，建议全部改为直接属性访问。这与 `train_stage1.py` 用 `_apply_argument_defaults` 处理编程式 Namespace 的场景不同，本文件不需要这层防御。
- `--fusion-mode` 的 choices 硬编码了一份与 `retrieval.FUSION_MODES` 完全相同的列表（`:682`），建议改为 `choices=FUSION_MODES`（`train_stage1.py:1817` 已经这么做了）。
- `_comparison_view` 的 `channel=None` 分支是死代码（`:260-266`），生产调用点全部显式传参，建议形参改为必填。
- `_teacher_feature_preflight` 里 `write_json` 被调用两次写同一份数据（`:136, 140`），`--preflight-only` 又把刚写完的 JSON 重新从磁盘读回来打印（`:432-439`），建议让该函数直接返回 payload 供复用。
- `payload["parameters"]` 手写展开了与 `aggregator.config()` 完全相同的 4 个字段（`:604-607`），建议直接用 `**aggregator.config()`。
- `_markdown` 函数里多处硬编码 `k=10`（`:297-338`），与可配置的 `--recall-ks` 存在冲突：一旦用户传入不含 10 的 ks 集合会在报告生成阶段 KeyError（此时 `metrics.json` 已经写完，只有 markdown 报告会崩），建议在参数校验阶段提前拦截。

### 6.3 库函数返回面过宽的信号（仅记录，不建议本轮改动）
- `evaluate_stage1_r3_baselines.py:198-216` 对每个 recall k 都调一次 `evaluate_teacher_reranking`，但只取其中一个字段；该库函数（`teacher_rerank.py`）每次都会顺带算出并丢弃 raw/teacher-only 的完整指标。这提示 `evaluate_teacher_reranking` 的返回契约可能过宽，但改动会影响其另一个调用方（`run_stage1_r3_sweeps.py`），建议留待后续单独评估。

---

## 七、优先级建议（综合各模块结论）

| 优先级 | 内容 | 预计收益 | 风险 |
|---|---|---|---|
| **P0：立即可做，零风险** | 各模块中标注的"确认死代码/纯转发"条目（`_channel_ranks`、`RECALL_KS`别名、`RawSearch` Protocol、`load_path_aggregation`、`multiplier`形参、`--train-data`/`--teacher-ensemble-alpha`死flag、`_comparison_view`的None分支、`preload_embedding_matrix`归零遗漏等） | 删除几百行从未执行的代码路径 | 极低，均已用 grep 全仓库确认无调用方 |
| **P1：结构性去重，收益大** | 1.1~1.4 节的聚合配置/checkpoint读写/import路径统一；2.3 节 training.py 的 dict 累加器与 train/dev 目标函数共享；3.2 节的重复函数抽取；6.1/6.2 节的 CLI 参数表收敛 | 预计可减少上千行重复代码 | 低，多数是纯提取/纯转发替换，需要跑一遍现有测试确认输出不变 |
| **P2：需要设计决策** | 1.7 节的 FusionConfig 引入；training.py 四个训练循环合一；checkpoint 相关模块边界重新划分（1.4） | 大幅降低长期维护成本 | 中，改动面大，建议放在其余项完成、测试稳定后单独排期 |
| **P3：需要用户先确认，不要擅自改** | 第零节列出的 10 条 | 修复潜在 bug | 会改变部分输出数值或报错行为，必须先确认设计意图 |

---

## 附：本次审查方法说明

本报告由 5 个并行子任务分别审查不同模块（分工：①`data/construction/mining`；②`models/objectives/training`；③`retrieval/evaluation/scoring/selection`；④`teacher_logits/teacher_rerank/features/pca`；⑤`train_stage1.py`/`evaluate_stage1_r3_baselines.py`）加上对小型工具模块（`artifacts.py`/`checkpoints.py`/`protocol.py`/`significance.py`/`workflow.py`/`evidence_diagnostics.py`）及两个数据缓存/挖矿 CLI 脚本的直接审查汇总而成。每一条"未使用/冗余"结论都通过 `grep` 在 `src/`、`tests/`、`scripts_old/` 全量交叉验证后才写入本报告，以降低误判概率；但由于代码量大，仍建议在实际改动前对每条建议做一次独立复核，并在改动后跑一遍 `tests/` 下对应的测试文件确认输出未变。
