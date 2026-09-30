# Stage-2 合并现状与复现缺口（MIGRATION_GAPS）

本文档记录 2026-09-29 那次 Stage-2 合并的**边界**：已经并进 `src/` 的是什么，尚未并入的是什么，
以及"能否用 `src/` 独立复现 B+IDF 那组数字"的逐项结论。

**一句话结论：方法配方（selector 训练 + 桥接逐行打分 + bridge-first reranking）已在 `src/` 中可运行；
本次补并后，C30 人口、R7 view seed、冻结 MiniLM backend 和 checkpoint-free B+IDF 融合也有了独立入口。
完整 9B 图像定位仍通过注入的 recovery backend 执行，训练浮点路径仍不承诺逐 bit 相同。**

本文档不宣称任何未完成的迁移已经完成。若后续补齐缺口，请更新本文档而不是删除它。

---

## 1. 本次已并入的内容

| 文件 | 作用 |
|---|---|
| `src/mmdd_stage2/natural_evidence.py` | 从指定 Stage-1 导出构造自然证据输入（`NAT_E`）+ source-group holdout 划分 + 覆盖 funnel |
| `src/mmdd_stage2/matched_row_score.py` | typed 值键规则 + D-0.98 "matched original rows / 5" + 独立逐行重算 |
| `src/mmdd_stage2/bridge_first_rerank.py` | bridge-first reranking：bridge tier → df/idf visible tier → Stage-1 tier，再 RRF60 |
| `src/mmdd_stage2/column_r2_training.py`（扩展） | 新增 `NAT_E` arm、`fixed_epoch` 受控模式、arm-aware 的 dev 子集证据源 |
| `src/run_stage2_columns_r2.py`（扩展） | CLI 暴露 `--kind nat_e/nat_e_test`、`--arm NAT_E`、`--fixed-epoch`、`build-natural-evidence` |
| `src/mmdd_stage2/minilm.py` | 本地冻结 MiniLM cell embedding backend |
| `src/mmdd_stage2/fresh_recovery_engine.py` | 不读取旧 selector checkpoint 的 Crop/Localizer/QueryRunner 边界 |
| `src/mmdd_stage2/b_plus_idf.py` / `src/run_stage2_b_plus_idf.py` | fresh bridge scores + 冻结 IDF visible scores 的融合入口 |
| `src/EVIDENCE_SELECTOR_AND_BRIDGE_RERANK.md` | 方法定义、可执行命令、数值性质与不可变约束 |

复用而非重写：reader 仍是 `mmdd_stage2/qwen.py`，head 仍是 `verifier.CandidateColumnScorer(head_type='mlp')`，
loss 仍是 `column_training.column_loss`，特征缓存仍是 `column_r2_cache.reader_worker`，
batch/view/权重机制仍是原有实现。`NAT_E` 是**新增一个证据来源**，不是新增一套架构。

### 1.1 已并入部分能复现什么

- fresh 训练的配方：同 seed 逐 tensor 相同初值（收据 `initial_parameter_sha256`）、
  固定 `--fixed-epoch 20`、`--early-stopping-patience 0`（dev 只监测不选优）、
  head 与 loss 与 R7 用的完全同一实现。
- 自然证据条件：`build-natural-evidence --stage1-dir <本轮 Stage-1 导出>`。
- 给定 bridges 与向量后可复现：逐行桥接打分（`bridge_row_scores` + 独立 `reference_bridge_row_scores` 交叉验证）、
  以及 bridge-first 三层排序与 RRF。

---

## 2. B+IDF 那组数字当时到底由什么算出（便于审计）

那次 addendum（R7 运行输出目录下的 `addendum/B_PLUS_IDF.py`）只读两处：

| 读入 | 来源 | 字段 |
|---|---|---|
| B 的桥接分数（唯一来自本 run 的科学输入） | 本 run `scores/SEL_E_FRESH/<split>/<q>.json` | `table_scores`（C30 范围，D-0.98 matched rows/5） |
| C50 候选顺序、可见层分数、生产桥接分 | 你的 IDF 实验 `LEXICO_IDF_ABLATION_V1/per_query_scores.json` | `base` / `vis_row` / `vis_idf` / `bridge` |
| 评价用 gold、人口标签、source_group | 同上 | `gold` / `kind` / `source_group` |
| STAGE1 列（B-only 表） | 本 run `stage1/<split>/<q>.json` | `C50` |

新 MiniLM 编码 **0 条**（`vis_row`/`vis_idf` 直接复用冻结值）。

事后补验的两个对齐事实（用于支撑"同候选域、同标签"的说法）：

- 两边 C50 候选顺序：**2364/2364 完全一致**
- 两边 gold 标签集合：**2364/2364 完全一致**

结论：B+IDF−IDF 确实是同候选域、同标签的对比，唯一变量是桥接分来源。现在可由
`src/run_stage2_b_plus_idf.py` 消费当前 fresh arm 的 bridge score；历史 R7 数字仍然只是在
提供历史 score artifacts 时可重放，不能把历史 artifact 当作 fresh 重新生成。

---

## 3. 训练范围澄清（避免"C50 训练"的误解）

| 项 | 事实 | 证据 |
|---|---|---|
| 训练 pair 范围 | **C30**，不是 C50 | funnel：7035 有标签 → 因落在 C30 外剔除 2656 → 4379 入训 |
| 全量校验 | 4379 个 train job 中，越出各自 query C30 的目标 = **0** | 逐 query 比对 `stage1/train/<q>.json` 的 `C30` |
| dev / test jobs | 每 query 恰 30 个（1198×30=35940；1166×30=34980） | `jobs/{dev,test}.jsonl` |
| selector 的输入形态 | 每对 `(query, target, E)` 一次 reader 前向；head 只给**该 target 的所有列**打 logit | `heads.py` / `column_loss` |

C50 只出现在三处，均与训练无关：Stage-1 导出的深度（取前 30 得 C30）、最终排名尾部 31–50 原样 append、
以及 IDF addendum 的**评估候选域**（你的 IDF 重排器本来就在 C50 上工作）。

推论：在 C50 评估域里，B 只能在前 30 名提供 bridge 分，31–50 落到 IDF 的 Tier2/Tier3。
这是 B+IDF 相对 IDF 只涨 0.05/0.34pp 的口径原因之一，不是模型本身弱。

---

## 4. 缺口清单

| # | 缺口 | 现状 | 影响 | 补法 | 风险 |
|---|---|---|---|---|---|
| 1 | **C30 监督范围** | 已由 `run_stage2_columns.py audit --candidate-scope-file` 支持目标表或列 scope | 用独立 scope 文件在人口层剔除 C30 外 qrels | 新 scope 必须写入新输出目录，不能与旧 R1 混用 |
| 2 | **view1 列排列 seed** | 已新增 `--view-seeds`；默认保持 `(13001, 29001, 47001)`，R7 可传 `13001 26002` | 自定义 seed 会进入 cache contract，且不会复用旧默认 view cache | 改动 seed 必须生成新的 feature manifest |
| 3 | **恢复/生成引擎** | `mmdd_stage2/fresh_recovery_engine.py` 提供 checkpoint-free `CropEngine`/`ImageLocalizer`/`QueryRunner` 边界，并复用当前 `SourceAwareRecoveryBackend` | 能从当前 Stage-1 任务执行顺序 fresh 生成；完整 R7 V-V localizer 作为注入 backend | 9B 运行仍需本地模型和授权 GPU，默认 localizer 是可审计的 original fallback |
| 4 | **MiniLM cell 向量供应** | `mmdd_stage2/minilm.py` 提供本地 `all-MiniLM-L6-v2` 冻结 backend，并实现 `embed_texts`/`vector`/`matrix` | D-0.98 TEXT↔TEXT 路径不再依赖外部 caller | 必须使用与历史缓存相同的本地权重和 pooling 配置 |
| 5 | **训练权重逐位一致** | `src/SELECTOR_TRAINING.md` 自述：FP32 batched 矩阵合并可能改低位浮点，**不承诺逐 bit 等于旧结果** | 即便 1–4 全补，也不能声称参数或指标逐位相同 | 接受"同配方、非逐位"的定位，或为复现单独走标量执行路径 | Adam 会放大末端公共 bias 的浮点残差，故不能只比对参数 hash |

---

## 5. 两条可选路线

### 5.1 路线 A：保持现状（推荐，零风险）

`src/run_stage2_b_plus_idf.py` 可以直接读取冻结的 IDF per-query 输入和当前 fresh
`scores/<arm>/<split>`，生成 B+IDF order 与 receipt。它不会读取 selector checkpoint，亦不会重新
编码可见层向量；若要重跑图像生成，先用 `fresh_recovery_engine.QueryRunner` 产出当前 arm 的
bridge scores。

### 5.2 可选：把完整 V-V localizer 迁入 `src/`

当前 `fresh_recovery_engine.py` 已提供顺序执行和 original fallback；若要把 R7 的完整
V-V/RAEA/Consensus localizer 代码也收进 `src/`，还需要解决两个前置问题：

1. **单卡硬绑。** `vendor/recovery/upstream/runtime/code/legacy_runtime/gpu_guard.py` 里
   `nvidia-smi -i 1` 与 `need(ix=='1' and uuid==c['gpu_uuid'])` 把物理 GPU1 写死，而外层包同时宣称
   支持 `--authorize-gpu0` / `gpu_physical_index`。二者矛盾：绑 GPU0 的 run 会在 recovery 阶段
   抛 `PHYSICAL_GPU1_IDENTITY`。迁移时必须先决定单一语义并改文档与实现。
2. **顺序与内存模型。** 引擎按"顺序子进程加载/退出"设计（同一张卡上先 reader、后生成、再释放），
   迁移时不要顺手引入并发，否则会改变显存行为与可复现性。

另外，迁移属于**功能变更**：需要新 run 命名空间，并如实重跑两臂，不能只补表现差的一臂。

---

## 6. 一条不可假定的数值性质

在本轮冻结的 entitables 人口上（用 R7 run 的 bridges 与 audits 的 MiniLM 缓存实测）：

- 匹配决策总数 32,462，其中 typed/normalized exact 10,507、TEXT 走 cosine 12,230、非 TEXT exact-only 7,950、无候选 1,775；
- 最终匹配 10,529 个，其中来自 MiniLM cosine 阈值路径的只有 **21–22 个（约 0.2%）**；
- cosine 分布 min 0.033 / mean 0.519 / max 1.000；**没有任何决策落在 0.98 阈值的 ±1e-5 内**（最近的在 1e-4 外）；
- 对向量做 `v' = normalize(v + εu)` 扫描，ε 到 1e-3（比已审计的单位范数偏差 1.788e-7 大约 5600 倍）
  仍未改变任何 table score、任何 RRF 顺序、R@10 Δ=+0.0000pp。

这条性质**只对本人口成立**，换人口必须重测，不得假定。它的实际用途是：说明在这个数据集上
"换 encoder / 换 device / 微调 τ" 都不会带来实质变化，真正能动的是**选哪些 (target, column) 去恢复**。

---

## 7. 现状验证

```bash
conda run -n MMDD python -m pytest $(grep -rl mmdd_stage2 tests/*.py) -q
# 197 passed（含本次新增 36 个用例）
```

行为保持的硬证据：新 helper（`prediction_evidence` / `training_condition` / `evidence_subset_ids`）
与改动前的原始表达式在 **6 个既有 arm × 22 项 = 132 个组合**上逐位一致，`NAT_E` 是唯一新增行为。

`tests/test_selector_evidence_arms.py::test_natural_evidence_arm_trains_without_any_checkpoint_or_prior_shortlist`
在全程监控下跑完 `NAT_E` 训练：一旦读取 `PRIOR` checkpoint 或调用 `freeze_prior_inputs` 即报错，
收据中 `prior_sha256=None`、`prior_frozen=False`。

---

## 8. 不要做的事

- 不要声称 `src/` 能复现 B+IDF 那组数字（§4 缺口未补）。
- 不要用旧 selector checkpoint 给训练数据打分或筛候选列；`NAT_E` 路径已明确禁止。
- 不要手工编辑 `COLUMN_POPULATION` / `INPUT_MANIFEST`：它们是 hash 锁定的数据合同，
  改动必须走 `run_stage2_columns.py audit` 并生成新目录。
- 不要把 `NAT_E` 与 `O-R` 当作可以互换的同一条件：`O-R` 是 R1 冻结检索的自然证据，
  `NAT_E` 是指定导出的自然证据；喂同一份导出时二者重合，喂不同导出时结果不同。
- 不要把本文件的缺口清单当作"已完成迁移"的说明。
