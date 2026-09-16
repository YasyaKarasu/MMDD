# EntiTables 数据集格式变更：共享数据湖

> 面向下游接入者的变更说明。完整格式规范见
> [`mm_joinability_downstream_data_format.zh-CN.md`](./mm_joinability_downstream_data_format.zh-CN.md)。

| | |
| --- | --- |
| 数据集 | `output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9` |
| 新 schema 版本 | `query-only-shared-data-lake-v1` |
| 迁移工具 | `src/migrate_query_only_split.py` |
| 迁移日期 | 2026-09-16 |

## 1. 契约

**切分只作用在 query 上。train/dev/test 共享同一片完整的数据湖。**

| | |
| --- | --- |
| 数据湖 | 22 886 张表，全体共享——三个 split 的检索语料完全相同 |
| Query | 14 994 条，按来源表切分：train 12 630 / dev 1 198 / test 1 166 |
| 切分键 | `source_table_id`——同一来源表派生的 query 永远同属一个 split |

旧格式把湖也切成了三份互不相交的子集（train 18 404 / dev 2 274 / test 2 208，两两交集
为 0，并集正好 22 886）。新格式没有这个概念了。

**test 的 query 会检索到由 train 来源表生成的湖表，这是预期行为，不是泄漏。** 泄漏边界在
query 侧：同一来源表的 query 不跨 split，模型在测试时从未见过该 query 的监督信号。湖是
被检索的对象，不是监督信号。

## 2. 产物清单

九个产物里两个的记录内容变了，没有新增产物。其余原封不动，包括全部多模态线索。

| 产物 | 记录数 | 变化 |
| --- | ---: | --- |
| `data_lake_tables/` | 22 886 | **移除 `split`**——13 525 条完整表 + 9 361 条 `source_table_ref` 引用，全部剥除 |
| `table_queryability_decisions.jsonl` | 20 000 | **移除 `split`** |
| `splits.json` | — | **整体重塑**，见 §3 |
| `dataset_manifest.json` | — | **新增契约字段**，见 §3 |
| `query_tables/` | 14 994 | 不变，`split` 保留 |
| `qrels.jsonl` | 15 870 | 不变，`split` 保留（继承自 query） |
| `evidence_recoveries/` | 23 592 | 不变，`split` 保留（继承自 query） |
| `source_tables/` | 20 000 | 不变，本来就没有 `split` |
| `bridge_assets/` | 225 720 | 不变 |
| `table_asset_links/` | 1 085 865 | 不变 |
| `entities/` | 348 781 | 不变 |
| `attribute_extractions/` | 722 268 | 不变 |

多模态线索层（`bridge_assets` / `table_asset_links` / `entities`）从来就是按 entity 组织的
全局资源，没有 split 作用域，这次一个字节都没动。

## 3. 逐字段 diff

### 湖表记录 · `data_lake_tables/part-00000.jsonl`

```diff
  "table_id": "target_07955d4252796693",
  "object_id": "target_07955d4252796693",
  "object_type": "table",
  "role": "target_data_lake_table",
- "split": "dev",
  "source_table_id": "st_table_0395_775_c9f42f053f",
  "page_title": "C-Bo",
  "columns": [...], "rows": [...]
```

`role = "raw_data_lake_table"` 的引用型记录同样剥除了 `split`。

### `splits.json` · 整体替换

```diff
- "train": {"source_table_ids": [...16031], "query_table_ids": [...12630],
-           "data_lake_table_ids": [...18404]},
- "dev":   {...1995, ...1198, "data_lake_table_ids": [...2274]},
- "test":  {...1974, ...1166, "data_lake_table_ids": [...2208]},
- "note":  "source-level split; data_lake contains generated targets ...",
  "split_key": "page_title_or_source_table_id",
+ "split_policy": "query_only",
+ "data_lake_scope": "shared",
+ "query_table_counts": {"train": 12630, "dev": 1198, "test": 1166},
+ "data_lake_table_count": 22886,
+ "data_lake_artifact": "data_lake_tables"
```

### `dataset_manifest.json` · 新增键

```diff
+ "split_schema_version": "query-only-shared-data-lake-v1",
+ "query_construction": {"split_policy": "query_only",
+                        "data_lake_scope": "shared", ...},
+ "published_single_files": {...}
```

`note` 重写为：*Train/dev/test assignments apply only to queries; every split retrieves
from the complete shared data lake.*

## 4. 你需要改什么

四条规则。如果你的代码本来就没按 split 过滤湖，那大概率一行都不用动。

**取 query 的 split：照旧读 `query_tables[*].split`。** 这个字段没变，而且现在是唯一来源。

**不要再读 `splits.json[split]["data_lake_table_ids"]`。** 连同 `source_table_ids` /
`query_table_ids` 一起没了。新的 `splits.json` 只是计数摘要，不再是 ID 清单。

**不要从湖表记录上读 `split`。** 该字段已不存在，`record.get("split")` 返回 `None`。如果
你有 `target["split"] == split` 之类的断言，删掉 target 那一侧，保留 query 侧仍然有意义。

> 注意这类断言的失败方式可能是**静默**的。如果它被包在 `except` 里当作「跳过该样本」，
> 你会得到一个空数据集而不是一个报错。我们在 `mmdd_stage2/column_data.py` 里就踩到了
> 这个：迁移后每一对都被判为 `object_split_mismatch` 丢弃，population 文件写成空的，
> 退出码仍是 0。

**检索语料 = 全湖。** 候选集和负例从全部 22 886 张表里采，不分 split；评测 dev/test 时
对着同一片湖排序。

## 5. 结果不会变

迁移是语义中性的。原因很简单：`split` 从来没有进入过表的序列化——
`serialize_table_parts()`、`visible_table()`、`serialize_table()` 都只读 `columns` 和
`rows`。所以嵌入、特征缓存和 `corpus_sha256` 全部不受影响。

用迁移前后的数据集分别跑同一套构建代码，逐字节对照：

- `stage1_corpus.jsonl` — 完全相同（它的哈希门控所有冻结的 ANN 索引与特征缓存）
- `stage1_objects.jsonl` — 完全相同
- `target_lists.jsonl` / `edge_lists.jsonl` — 完全相同
- `COLUMN_POPULATION` / `COLUMN_INPUTS` 的 train、dev、test 六个文件 — 与已冻结的产物逐字节相同
- `OBJECTS.jsonl.gz` — 解压后内容相同（gzip 头含时间戳，字节层面必然不同）

已实跑验证的消费方：`mmdd_stage1/construction.py`、`mmdd_stage2/column_data.py`、
`mmdd_stage2/oracle.py`、`mmdd_stage1/evidence_diagnostics.py`、
`audit_stage1_r10_protocol.py`。

## 6. 判断格式 / 复现迁移

manifest 是权威来源。用 `mmdd_dataset.wdc_runtime.iter_dataset_artifact()` 读产物，它会
按 manifest 解析相对路径。

```python
# 新格式当且仅当这一项存在
manifest["split_schema_version"] == "query-only-shared-data-lake-v1"

# 等价的运行时判据
all("split" not in r for r in iter_dataset_artifact(root, "data_lake_tables"))
```

仓库里的构造器（`scripts_old/build_mm_joinability_dataset.py` 等）现在直接产出新格式，
重新构建不需要迁移。迁移工具只用于把**已有的**旧格式产物原地改写（非幂等）：

```bash
conda run -n MMDD python src/migrate_query_only_split.py --dataset-root <dataset_dir>
```

## 附：一处实现细节

只在你要写 manifest 校验器时才相关。这份数据集的 shard 条目只有 `path` 和 `records`，
没有 `bytes` / `sha256`。这是刻意保持的：`iter_dataset_artifact()` 一旦发现校验和，就会
在吐出第一条记录前对整个产物做一次 SHA-256，而 `data_lake_tables` 有 260 MB 且有多个热
读点。迁移工具会跟随数据集原有的详细度，不擅自添加；对本身带校验和的数据集则照常更新。

迁移前的 `data_lake_tables/`、`table_queryability_decisions.jsonl`、`splits.json`、
`dataset_manifest.json` 已备份在数据集同级的 `*.pre_query_only/` 目录。
