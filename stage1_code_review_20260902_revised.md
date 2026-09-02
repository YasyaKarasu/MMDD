# Stage-1 代码审查报告：修订版

日期：2026-09-02  
审阅范围：`src/mmdd_stage1/` 全部模块及主要 CLI 入口

**验证状态**：原报告中的核心问题（Z1-Z10 以及死代码判断）已逐一验证，下文标注了验证结果。

---

## 零、需要用户先拍板的疑似 Bug / 设计冲突（✅ = 已验证确实存在）

| # | 位置 | 问题 | 验证 | 建议 |
|---|---|---|------|------|
| Z1 | `objectives.py:102-105` vs `retrieval.py:577-579` | 训练期 comb_mnz 用 `evidence_mask.sum`（有效证据数），检索期用 `value != 0.0` 计数（非零分数数）。当路径分恰为 0 时两者不同。 | ✅ | **必须统一**。建议改检索侧为 `len(evidence_scores)`，与训练侧语义对齐（都是"参与聚合的证据数"） |
| Z2 | `teacher_logits.py:44-50` vs `:456-461` | 缓存文件名只在特定 aggregation 下编码 temperature/power，但加载时对**所有**配置都严格校验 4 字段，导致本应命中的缓存抛硬错。 | ✅ | **修改加载校验逻辑**：只校验文件名中实际编码的字段。或者统一把 4 字段全编入文件名（会导致缓存失效） |
| Z3 | `train_stage1.py:1249` vs `:1280` | teacher-edge 用 `or 1` 强制覆盖，teacher-path 直接传参沿用 checkpoint。CLI help 描述的是后者行为，与前者矛盾。 | ✅ | **统一行为**。建议 teacher-edge 也改为直接传参（删除 `or 1`），让两个 stage 一致，且符合 help 文档 |
| Z4 | `train_stage1.py:1188-1191` | `--primary-metric` 默认 `"recall@10"`，但 teacher-rerank 模式的 gate_metrics 只有带作用域前缀的 key（如 `teacher_rerank.recall@10`），默认值必然 KeyError。 | ✅ | **修改默认值**为 `"teacher_rerank.recall@10"`，或在参数校验阶段提前拦截不合法的 metric 名 |
| Z5 | `train_stage1.py:111-124` | student 侧支持扁平/嵌套 key 两种取法，raw 侧只支持嵌套。扁平 key 会导致 raw 侧 per-query 恒为 None，bootstrap CI 静默退化成点估计。 | ✅ | **加校验或告警**：当 metric 不含 `.` 且需要 bootstrap 时，至少 log warning |
| Z6 | `data.py:250` | `"teacher_edge_ensemble_alpha"` key 全仓库无生产者写入，是死键；而 `refresh_stage1_hard_negatives.py:262` 写入的通用 `"teacher_ensemble_alpha"` 在当前代码路径下永远读不到。 | ✅ | **删除死键**，修复回退链使其能读到通用键 |
| Z7 | `construction.py:492` | `for evidence_type in {asset_types[e] for e in positive_evidence_ids}:` 用 set 字面量迭代驱动随机选择，迭代序依赖 PYTHONHASHSEED，同一 seed 在不同进程下可能选出不同 fallback evidence。 | ✅ | **改为 `sorted(set(...))`**，保证可复现性 |
| Z8 | `retrieval.py:739-778` | `fuse_ranked_channels` **原地修改** `result["score"]`。`evaluation.py` 对同一批对象调用 3 次，后续调用覆盖前面的值。当前下游只读 target_id/paths 所以未暴露，但潜在隐患。 | ✅ | **修改为返回新 dict** 或深拷贝输入，避免跨调用耦合 |
| Z9 | `construction.py:376` vs `:443-445` | 两处看似同构的回退表达式，key 取值范围其实不同（前者可能命中 recovery_evidence 里的非 positive 条目），不是真正等价。 | ⚠️ | **保留现状**，但建议加注释说明两者的差异，避免未来被误合并 |
| Z10 | `retrieval_aligned.py:160-161` vs `align_edge_record` | target 有配额溢出校验，edge 无。edge 超额时会静默产出长度超限的候选列表。 | ⚠️ | **补齐 edge 侧校验**，或确认 edge 侧有意允许超额 |

**行动建议**：
- **Z1/Z2/Z3/Z4/Z7 优先修复**（影响数值正确性或必然崩溃）
- Z5/Z6/Z8 次优先（潜在隐患）
- Z9/Z10 可稍后处理（需确认设计意图）

---

## 一、跨模块的结构性问题（高优先级，大幅减少重复代码）

### 1.1 PathAggregator 聚合公式三处独立实现
**问题**：`objectives.py:53-107`（训练期张量版）、`retrieval.py:532-580`（检索期标量版）分别实现 7 种聚合，已出现分叉（Z1）。

**建议**：
1. **短期**：在两处加交叉引用注释，提醒维护者同步修改
2. **长期**：把标量版重写为张量版的参考实现，用测试锁定一致性

### 1.2 PathAggregator 的 4 字段被 6+ 处手写比较/重建
**问题**：`training.py:657-670`、`train_stage1.py:447-454`、`teacher_logits.py:35-41`、`data.py:320-329`、`evaluate_stage1_r3_baselines.py:604-607`、`checkpoints.py:56-77` 各自手写相同的 4 字段展开/比较逻辑。

**建议**（可直接落地，零行为变化）：
```python
# 替换所有手工构造为：
TeacherScoreConfig(**aggregator.config())

# data.py:320-329 直接简化为：
raw_score_config = metadata  # 字段与默认值逐位相同

# evaluate_stage1_r3_baselines.py:604-607 改为：
**aggregator.config(),

# training.py:660-666 的逐字段比较改为 dataclass __eq__：
if old_config != TeacherScoreConfig(**aggregator.config()):
    ...

# 删除 checkpoints.py 的死代码 load_path_aggregation
```

### 1.3 import 路径混乱：checkpoint_fingerprint / write_json 各有两条路径
**问题**：
- `checkpoint_fingerprint` 真实定义在 `artifacts.py`，但 `retrieval.py:16` 转导出，约 25 个脚本从 `retrieval` 导入
- `write_json` 定义在 `artifacts.py`，`selection.py:13` 转导出，约 15 个脚本从 `selection` 导入
- 另一批脚本直接从 `artifacts` 导入

**建议**（纯 import 路径调整，改动面大，收尾阶段一次性做）：
```bash
# 统一所有 import 为：
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json

# 删除 retrieval.py 和 selection.py 的转导出
```

### 1.4 checkpoint 读写分离，职责散落三处
**问题**：
- `training.py:1221-1236` 持有 `checkpoint()` 序列化逻辑
- `checkpoints.py` 持有 `load_*` 反序列化系列
- `selection.py` 又持有 `CheckpointManager`（保存/拷贝/剪枝）

**建议**（需同步改 import）：
1. 把 `training.py` 的 `checkpoint()` 移到 `checkpoints.py`
2. 把 `selection.py` 的文件管理逻辑也移到 `checkpoints.py`
3. `selection.py` 只保留 gating 相关逻辑（`MetricGate`/`GateDecision` 等）

### 1.5 JSONL record schema 有 3 个独立写入者、1 个解析者
**问题**：
- 写：`construction.py:456-467/551-561`、`mining.py:250-285`、`retrieval_aligned.py:222-231`
- 读：`data.py:207-393`
- 已出现不一致：`construction.py` 无条件写 `"split"`，`mining.py` 仅非 None 时写

**建议**：
```python
# 在 data.py 新增公开函数（读写成对）：
def target_record(example: TargetExample) -> dict[str, Any]:
    ...
def edge_record(example: EdgeExample) -> dict[str, Any]:
    ...

# mining.py 改为导入使用
# construction.py/retrieval_aligned.py 至少共用字段名常量
```

### 1.6 CLI 通用逻辑在多个脚本里各写一份
**重复逻辑举例**：
- `--evidence-modality-weights` 解析：4 份逐字相同
- `--recall-ks` 解析：4 份不同实现（库里有 `_validate_recall_ks` 却不复用）
- PathAggregator "CLI 覆盖或沿用 checkpoint" 20+ 行 × 2
- device 解析三行式：13 个脚本重复
- dev 样本加载：5 份
- ensemble alpha 校验 `[0,1]`：4 份

**建议**：
在 `mmdd_stage1` 新增工具模块（或复用 `protocol.py`），收敛为公开函数供 CLI import。

### 1.7 检索参数透传 4~5 层，同一组 ~20 参数抄了 5 遍
**问题**：
- `retrieve_zero_one_hop` → `retrieve_zero_one_hop_detailed` → `retrieve_zero_one_hop_detailed_many`
- `evaluation.py:146-174` 又抄一遍
- 默认值分叉：retrieval 侧 `fusion_mode="rrf"`，evaluation 侧 `"weighted_rrf"`

**建议（保守）**：
1. 删除 `retrieve_zero_one_hop_detailed`（纯包装，52 行参数表）
2. `..._detailed_many` 内部递归改为循环，避免 kwargs 重复抄写

**建议（彻底）**：
引入 `FusionConfig` frozen dataclass 承载参数组，各层只传一个对象。

---

## 二、models.py / objectives.py / training.py

### 2.1 models.py

**死代码/冗余**：
1. ✅ **`"orthogonal"` 与 `"random_orthogonal"` 完全相同的代码路径**  
   建议：合并为一个常量，或直接从 `STUDENT_INITIALIZATIONS` 移除未使用的
   
2. ✅ **`compress()` 是 `compress_many()` 的单对象重复**（200 行 vs 320 行）  
   生产代码无调用点。建议：改为对 `compress_many()` 的薄委托

3. **relation 打分公式被写了 4 遍**  
   `score_embeddings`、`score_embedding_matrix`、`score_pairs` 内联、`relation_query`/`index_vector`  
   建议：抽 `_score_projected` / `_score_projected_matrix` 私有方法

4. **`IdentityStudentJoinabilityModel` 与 `ProjectedIdentityStudentJoinabilityModel` 逐字重复**  
   `relation_query`/`index_vector` 逻辑。建议：抽 mixin 共享

**小问题**：
- relation 初始化噪声硬编码 `0.01`，与可配置的 `initialization_noise_std` 脱节
- `freeze_projections` 被赋值两次（`:492` 与 `:528`）
- `_compression_buckets` 的 `max_padded_tokens` 形参从未被覆盖，可提为常量

### 2.2 objectives.py

1. **PathAggregator 继承 `nn.Module` 但没有参数/缓冲**  
   全仓库无 `.to()/.state_dict()` 调用，纯粹为了调用语法  
   建议：改为 `@dataclass(frozen=True)` + `__call__`，天然获得值相等语义

2. **`positive_indices` 与 `positive_mask` 重复解析**  
   `optional_listwise_cross_entropy` 内部对同一批数据解析两次 mask

3. **comb_mnz 分支靠隐式 else 兜底**（`:101-105`）  
   建议：改成显式 `elif` + 最终 `else: raise AssertionError`

### 2.3 training.py（问题最集中的文件）

**结构性冗余（收益最大）**：

1. **四个 `train_*` 函数共享同一套 epoch 脚手架**（`:737-1218`）  
   包括 rng 播种、`sample_mixed_epoch`、进度条、`_epoch_record`/`_finish_epoch`  
   **建议**：提取 `_run_training_loop(model, ..., step_fn, dev_fn)` 通用骨架  
   这是收益最大但改动面也最大的一项，建议单独排期

2. **每个指标一对 `pending_/values` 列表的样板重复 9 次**（`:1055-1194`）  
   建议：改为 `dict[str, list[...]]` 累加器统一处理，约省 70 行

3. **dev 评估函数与训练循环体逐字重复** 25~40 行  
   `_student_edge_objective` vs `train_student_edges` 内联块  
   `_student_path_objective` vs `train_student_paths` 内联块  
   建议：抽共享的 `_edge_batch_objective`/`_path_batch_objective`

**确认死代码**：
- ✅ `sample_balanced_epoch`（`:42-64`）生产路径无调用
- ✅ `_list_scores` 的 `positive_mask` 形参从未被传参
- `_target_teacher_scores` 白算两次 `target_positive_mask` 但下游只读 `.logits`

**小问题**：
- `_list_scores` 内联重复实现了 `scoring._mask`
- `student_relation_drift`/`student_projection_drift` 是纯模型自省逻辑，应在 `models.py`
- 训练函数默认值不统一

---

## 三、retrieval.py / retrieval_aligned.py / evaluation.py / scoring.py

### 3.1 确认死代码

✅ `retrieval.py:583` **`_channel_ranks()` 全仓库无调用点**，直接删除  
✅ `retrieval_aligned.py:11-14` **`RawSearch` Protocol 无实际使用**，删除  
✅ `teacher_rerank.py:19` `RECALL_KS = DEFAULT_RECALL_KS` 纯别名  
`evaluation.py:197-206` 不做实质工作的 progress 条

### 3.2 重复实现同一逻辑

1. **z-score 计算**：`retrieval.py:607-613` 与 `:917-938` 各写一遍  
   建议：抽 `_zscore_stats`

2. **`StudentANNIndices` 与 `RawEmbeddingANNIndices` 的 HNSW 索引加载重复** 40 行  
   建议：抽共享函数

3. **`scoring.py` 的 `_mask()` 与 `training.py` 的 candidate mask 构造逐字重复**  
   建议：`training.py` 直接复用 `scoring` 的函数

4. **`evaluation.py:65-84` 与 `teacher_rerank.py:86-136` 的指标聚合重复**  
   且求均值写法不统一（`sum/len` vs `statistics.fmean`）

### 3.3 冗余计算

- `fused` 通道指标被完整计算两遍（`:122` 与 `:125`），后者只取 coverage 字段
- `_channel_metrics` 对每个 k 重新计算一次 MRR 却只取 recall
- `positive_evidence_hits` 对所有 k 都计算但只有 `k=10` 被读取

---

## 四、teacher_logits.py / teacher_rerank.py / features.py / pca.py

### 4.1 teacher_logits.py

1. `_raw_edge_scores` 构造的 `candidate_mask`/`positive_indices` 从未被下游消费，属死计算
2. `score_and_cache_cosine_logits` 借道 ensemble API（`alpha=0`）表达"纯 z-score"，可读性差
3. 两个 `score_and_cache_*` 函数尾部写回逻辑大段重复
4. `tuple(float(v) for v in X[row, :count].cpu().tolist())` 模式出现 8 次
5. `teacher_logit_mode` 是完全可从 `ensemble_alpha` 派生的冗余字段（此项改动影响 schema，需确认）

### 4.2 teacher_rerank.py

1. `_teacher_scores` 用"key 类型三元分支"处理两种模式，建议统一成 `cache = score_cache or {}`
2. "按分数排序取 ranking" 重复 3 处，"overall + by_dataset" 三段式重复 3 处
3. `delta_k` 的动态 key 选择是无效通用化，下游都硬编码读 `"recall@10_delta"`
4. `_retrieval_metrics` 与 `evaluation.py:84` 同名不同义，建议改名消歧

### 4.3 features.py

1. `estimated_feature_bytes` 两个分支单位口径不一致（张量实际字节 vs 磁盘文件大小）
2. `preload_embedding_matrix` 空列表早退分支漏了 `self._cache_bytes = 0` 归零
3. `from_path` 的 eager 分支丢弃了 `cache_size` 参数
4. `configure_hot_cache` 与 `cache_info()` 用两组不同键名报告同一组指标

### 4.4 pca.py

1. 两套 PCA 实现（随机化 vs 精确），`explained_variance_ratio` 同名不同义
2. **隐性耦合**：`PCA_FORMAT_VERSION` 与 `PCA_SPECTRUM_FORMAT_VERSION` 数值恰好相同（都是 1），任一方 bump 会导致 spectrum artifact 无法加载
3. `mean` 字段被计算、持久化，但全仓库无读取点，属死输出

---

## 五、train_stage1.py / evaluate_stage1_r3_baselines.py 内部冗余

### 5.1 train_stage1.py

1. `_referenced_object_ids` 与 `_object_ids` 是同一函数的两个副本
2. `preload_embeddings` 代码块在 if/elif 两分支逐字重复
3. `_EpochController.__call__` 两次 `evaluate_student_retrieval` 调用 22 参数中 20 个相同
4. `MetricGate` 的内部状态被 `_EpochController` 直接改写，与 `selection.py` 的 `observe` 并存
5. `_apply_argument_defaults` 被调用 3 次且每次重建 90 参数的 ArgumentParser，前两次是 no-op
6. ✅ **`--train-data` legacy flag 全仓库无非 None 调用**，连同分支一并删除
7. ✅ **`--teacher-ensemble-alpha` 弃用别名无调用**，删除

### 5.2 evaluate_stage1_r3_baselines.py

1. 13 处 `getattr(args, key, default)` 全是死防御（args 只来自本文件 parse_args）
2. `--fusion-mode` 的 choices 硬编码列表，应改为 `choices=FUSION_MODES`
3. `_comparison_view` 的 `channel=None` 分支是死代码
4. `_teacher_feature_preflight` 写 JSON 两次，`--preflight-only` 又从磁盘读回
5. `_markdown` 函数多处硬编码 `k=10`，与可配置的 `--recall-ks` 冲突

---

## 六、优先级建议

| 优先级 | 内容 | 预计工作量 | 风险 |
|---|---|---|---|
| **P0（立即修复，影响正确性）** | Z1/Z2/Z3/Z4/Z7（必然崩溃或数值错误） | 2-4 小时 | 极低，纯 bug 修复 |
| **P1（死代码清理，零风险）** | `_channel_ranks`、`RawSearch`、`load_path_aggregation`、`--train-data`、orthogonal 合并等 | 1-2 小时 | 极低，已用 grep 确认 |
| **P2（结构性去重，收益大）** | 1.1~1.6 节、2.3 节 dict 累加器、3.2 节重复函数 | 1-2 天 | 低，需跑测试确认输出不变 |
| **P3（大重构，单独排期）** | 1.7 FusionConfig、training.py 四训练循环合一、1.4 checkpoint 模块重组 | 3-5 天 | 中，改动面大 |
| **P4（需设计决策）** | Z5/Z6/Z8/Z9/Z10、teacher_logit_mode 冗余字段、PCA schema | 按需 | 可能影响外部消费者 |

---

## 七、立即可落地的快速修复（Zero-Risk Quick Wins）

这些改动可以立即执行，零风险，直接删除死代码：

```python
# 1. 删除 retrieval.py:583-588 _channel_ranks
# 2. 删除 retrieval_aligned.py:11-14 RawSearch Protocol
# 3. 删除 checkpoints.py:56-62 load_path_aggregation（改测试为断言 load_path_aggregator 的属性）
# 4. 删除 train_stage1.py 的 --train-data flag 及相关分支
# 5. 删除 train_stage1.py 的 --teacher-ensemble-alpha 弃用别名
# 6. 合并 STUDENT_INITIALIZATIONS 中的 orthogonal/random_orthogonal
# 7. 修复 construction.py:492 的 set 迭代为 sorted(set(...))
```

---

## 附：验证方法说明

本次修订对原报告的关键问题（Z1-Z10）和主要死代码判断进行了逐一代码验证：
- 读取了相关代码片段确认问题存在
- 用 `grep` 确认调用关系和死代码判断
- 标注了 ✅（已验证确实存在）、⚠️（需进一步确认设计意图）

**建议落地流程**：
1. 先处理 P0（必须修复的 bug）
2. 再清理 P1（死代码）
3. 逐步推进 P2（结构性去重）
4. P3 单独排期评估
5. P4 与用户讨论设计意图后再决定
