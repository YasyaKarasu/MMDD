# EntiTables / WDC Joinability 下游数据格式

本文描述 `build_mm_joinability_dataset.py`、`build_wdc_mm_joinability_dataset.py`
以及 WDC 200K materializer 当前共同产出的 canonical multimodal joinability
格式。目标读者是实现检索模型训练、负例采样、离线评测和多模态 evidence
训练的开发者。

本文只描述 builder 的 canonical 输出。`output_stage1_logic` 是从这些数据进一步
派生的另一套 Stage-1 格式，字段名并不相同，详见“与现有 Stage-1 格式的关系”。

## 1. 最小数据依赖

直接进行 query-to-table 检索训练或评测时，需要读取：

| 文件或 artifact | 用途 | 是否必需 |
| --- | --- | --- |
| `dataset_manifest.json` | 定位本次构建有效的 shard | 必需 |
| `query_tables/` | 检索 query | 必需 |
| `data_lake_tables/` | 候选 target 表 | 必需 |
| `source_tables/` | 展开轻量的 raw data-lake 引用 | 必需 |
| `qrels.jsonl` | query-target 正例与相关度 | 必需 |
| `splits.json` | train/dev/test 的 ID 列表 | 必需 |

训练或评测多模态 evidence 路径时，另外读取：

| artifact | 用途 |
| --- | --- |
| `evidence_recoveries/` | query row → evidence → target row 的监督路径 |
| `bridge_assets/` | 路径引用的网页/Wikipedia 文本或图片 |
| `attribute_extractions/` | 模型从素材中抽取出的候选属性 |
| `entities/` | entity 标识、别名和来源位置 |
| `table_asset_links/` | source table cell、entity 和素材之间的关联 |

`stats.json`、`table_queryability_decisions.jsonl` 和 WDC 的 failure JSONL
主要用于构建审计与错误分析，不应作为模型输入。

### 1.1 下游是否需要读取 cache

需要按任务区分：

| 下游任务 | 是否依赖 cache | 原因 |
| --- | --- | --- |
| 纯 query-to-table 训练/评测 | 否 | 表格、qrels、split 都已写入 dataset output |
| 文本 evidence 训练/评测 | 否 | 文本正文已写入 `bridge_assets[].content` |
| 图片 evidence 训练/评测 | **是，当前需要图片目录** | `bridge_assets` 只保存图片路径和元数据，图片二进制仍在 cache image directory |
| 断点续跑或重新构建数据集 | 是 | 页面、下载和模型 cache 用于恢复构建 |

这里的“需要 cache”不等于下游要读取整个构建缓存。下游不应读取 Wikipedia/page
cache、WDC SQLite、model extraction cache 或 auto-check cache 来补充 JSON 记录；
canonical shard 已经包含训练和评测需要的结构化记录。图片是当前唯一需要额外保留的
大文件依赖，常见位置为：

- EntiTables：`<cache_dir>/images/`；
- 直接 WDC builder：`<cache_dir>/wdc_images/`；
- WDC 200K staged pipeline：`<cache_dir>/images/`。

实际路径必须以每条 image asset 的 `local_path` 为准。`dataset_manifest.json` 中的
`cache` 或 `web_cache` 段是构建 provenance，不是下游扫描 cache 的接口。

因此，当前 dataset output 对表格和文本任务是自包含的；对图片任务是“记录
自包含、图片文件外置”。如果数据要迁移到另一台机器，应同时迁移图片目录并保持
`local_path` 可解析，或者通过正式的打包步骤把图片复制/硬链接到
`<dataset_root>/images/`，同步更新 image asset 的 `local_path`、`relative_path` 和
相应 manifest 校验信息。不要只复制 output 目录后删除 cache。

典型目录如下：

```text
<dataset_root>/
├── dataset_manifest.json
├── qrels.jsonl
├── splits.json
├── stats.json
├── table_queryability_decisions.jsonl
├── source_tables/
├── query_tables/
├── data_lake_tables/
├── entities/
├── bridge_assets/
├── table_asset_links/
├── attribute_extractions/
└── evidence_recoveries/
```

## 2. 通用读取规则

1. 所有 shard 都是 UTF-8 JSONL，每一非空行是一个完整 JSON object。
2. 必须按 `dataset_manifest.json -> artifacts -> <artifact> -> shards` 中列出的
   `path` 顺序读取，不要直接 glob 目录。复用输出目录时，目录中可能残留不属于
   当前构建的旧 shard。
3. 大规模数据应逐行流式读取；不要一次性加载全部 WDC 表或素材。
4. ID 应作为不透明字符串使用，不要从 `query_`、`target_`、`chain_` 等前缀
   反推出业务语义。
5. 表格模型的主要可见输入是 `columns[].column_name` 和
   `rows[].cells[].text`。`raw` 可能是字符串、数组或 object，不保证适合直接送入
   tokenizer。
6. `source_column_index`、`source_row_id` 和各种 provenance 字段只用于追踪原表，
   不代表投影后表格中的局部位置。

仓库内的推荐读取入口是：

```python
from pathlib import Path
from stage1_io import iter_manifest_records

root = Path("output_mm_joinability_v15")

queries = iter_manifest_records(root, "query_tables")
targets = iter_manifest_records(root, "data_lake_tables")
```

运行独立脚本时需要把仓库的 `scripts/` 放入 `PYTHONPATH`。这里推荐
`iter_manifest_records()`，因为它既按 manifest 读取 shard，也会自动展开
`data_lake_tables` 中的 `source_table_ref`。

## 3. 表格的基础结构

### 3.1 Column

```json
{
  "column_index": 0,
  "source_column_index": 2,
  "column_name": "Year"
}
```

| 字段 | 含义 |
| --- | --- |
| `column_index` | 当前投影表中的零基局部列号 |
| `source_column_index` | 该列在 `source_tables` 原表中的列号 |
| `column_name` | 规范化后的可见列名 |

`source_tables` 中的 column 通常没有 `source_column_index`，并可能额外包含
`is_numeric_column` 等 profile 信息。

### 3.2 Row 和 Cell

```json
{
  "row_id": 0,
  "source_row_id": 17,
  "cells": [
    {
      "column_index": 0,
      "source_column_index": 2,
      "column_name": "Year",
      "text": "1993",
      "raw": "1993",
      "wiki_title": null,
      "has_wiki_link": false
    }
  ]
}
```

| 字段 | 含义 |
| --- | --- |
| `row_id` | 当前投影表中的零基局部行号 |
| `source_row_id` | 对应 source table 的稳定行号 |
| `cells[].column_index` | 当前投影表中的局部列号 |
| `cells[].text` | 下游编码应优先使用的规范化文本 |
| `cells[].raw` | 原始值；类型不固定 |
| `cells[].wiki_title` | EntiTables 的 Wikipedia 标题或 WDC 的内部 entity alias，可为空 |
| `cells[].has_wiki_link` | 当前 cell 是否被当作 entity cell |

对 query/target 表，`columns[i].column_index == i`，同一行中对应 cell 的
`column_index` 也应为 `i`。原始位置必须通过 `source_column_index` 和
`source_row_id` 关联，不能把局部索引当作原始索引。

## 4. `source_tables`

每条记录是一张完整的规范化源表，核心字段为：

| 字段 | 含义 |
| --- | --- |
| `source_table_id` | source table 主键 |
| `source_file` | 输入文件相对路径或来源标识 |
| `page_title`、`caption`、`section_title` | 表格外部文本上下文 |
| `num_rows`、`num_cols` | 完整源表大小 |
| `columns`、`rows` | 完整表格内容 |
| `metadata.candidate_entity_columns` | 候选 entity 列的 source column index |
| `metadata.column_profiles` | 非空率、唯一率、数值率等构建期统计 |

WDC 会把输入的 `image` 属性用于寻找素材，但不会把它写入 source、query 或
target 的 `columns`。WDC source table 还会带有
`provenance_builder = "build_wdc_mm_joinability_dataset.py"`。

## 5. `query_tables`

query table 的通用字段如下：

| 字段 | 含义 |
| --- | --- |
| `table_id` / `object_id` | query 主键；两者当前相同 |
| `object_type` | 固定为 `table` |
| `role` | 固定为 `query` |
| `split` | `train`、`dev` 或 `test` |
| `source_table_id` | 产生该 query 的 source table |
| `columns`、`rows` | 模型实际可见的 query 内容 |
| `source_column_indices` | 可见列在 source table 中的列号 |
| `source_row_indices` | 当前 row view 使用的 source row ID |
| `chain_id` | query-target join chain 的标识 |
| `row_view_index` | 同一 train chain 的第几个不相交 row view |
| `query_entity_col` | entity 列在 source table 中的列号 |
| `query_entity_col_name` | entity 列名 |
| `target_table_ids` | 构建期正例 target ID；只能用于监督，不能编码进 query |
| `hidden_attributes` | 隐式 query 被隐藏的 join 属性；只能用于监督和分析 |

### 5.1 隐式多模态 query

当对应 qrel 的 `reason` 为 `model_recoverable_join_column` 时：

- join 属性不出现在 query 的 `columns` 中；
- `hidden_attributes` 恰好描述被隐藏并可从多模态素材恢复的属性；
- 当前策略保证每个可见 query 只有一个 target qrel；
- `evidence_recoveries` 可以为 query 的各行提供恢复路径。

### 5.2 显式可见 join query

当 qrel 的 `reason` 为 `explicit_visible_join_column` 时：

- query 和 target 都能直接看到 join 列；
- query 会带 `construction_type = "explicit_visible_join"`；
- `hidden_attributes` 为空；
- `join_col` 和 `join_col_name` 描述可见 join 列；
- 这种 query 通常没有多模态 evidence recovery。

训练和评测时建议分别报告 implicit 与 explicit 指标，避免显式 join 的较低难度
掩盖隐式多模态检索效果。

## 6. `data_lake_tables`

候选集合中有两种记录。

### 6.1 可检索 target table

`role = "target_data_lake_table"` 的记录包含完整投影表：

| 字段 | 含义 |
| --- | --- |
| `table_id` | candidate 主键，也是 qrel 的 `target_table_id` |
| `split` | candidate 所属 split |
| `columns`、`rows` | target 的可见内容 |
| `join_col`、`join_col_name` | join 列在 source table 中的位置和名称 |
| `queryable_source_table` | 当前为 `true` |
| `target_context_col_names` | target 侧额外上下文列 |

target 通常保留列投影后所有满足最低要求的 source rows，而 query 只包含选中的
row view。因此同一 chain 应满足：

```text
set(query.source_row_indices) ⊆ set(target.source_row_indices)
```

### 6.2 Raw data-lake table 引用

未产生 query 的 source table 仍可进入候选库。为了避免重复存储，它使用轻量引用：

```json
{
  "table_id": "dl_raw_st_example",
  "role": "raw_data_lake_table",
  "split": "test",
  "source_table_id": "st_example",
  "source_table_ref": {
    "artifact": "source_tables",
    "source_table_id": "st_example"
  },
  "queryable": false,
  "reason": "no_column_met_recovered_value_ratio"
}
```

物理记录没有 `columns` 和 `rows`。下游不得把它当成空表；必须通过
`source_table_ref` 找到 source table 并展开成逻辑 data-lake table。仓库的
`stage1_io.iter_manifest_records(root, "data_lake_tables")` 已实现该逻辑。

## 7. `qrels.jsonl`

每条 qrel 是一个 query-target 正例：

```json
{
  "query_table_id": "query_example",
  "target_table_id": "target_example",
  "data_lake_table_id": "target_example",
  "rel": 3,
  "split": "test",
  "chain_id": "chain_example",
  "row_view_index": 0,
  "source_table_id": "st_example",
  "join_attribute": {
    "source_column_index": 2,
    "column_name": "Year",
    "role": "model_recoverable_join_column"
  },
  "reason": "model_recoverable_join_column"
}
```

字段约束：

- `target_table_id` 与 `data_lake_table_id` 当前相同；新代码应优先使用
  `target_table_id`，读取旧数据时可回退到 `data_lake_table_id`。
- `rel = 3` 表示强相关正例。训练负例不是以 `rel = 0` 写入，而是从同 split
  candidate corpus 中动态采样。
- `reason` 当前主要有 `model_recoverable_join_column` 和
  `explicit_visible_join_column`。
- 当前 `keep_best_recovery_single_target` 策略要求每个隐式
  `query_table_id` 恰好只有一条 qrel。
- `join_attribute` 是标签与审计信息，不能序列化到普通 table-retrieval query
  encoder 的输入中。

## 8. 多模态 evidence 格式

### 8.1 `bridge_assets`

所有素材都以 `asset_id` 为主键，并通过 `asset_type` 区分：

- `asset_type = "text"`：正文在 `content`，可能带
  `text_chunk_index`、`text_chunk_count`、`url`、`page_url`。
- `asset_type = "image"`：图片位置在 `local_path` 或 `relative_path`，并可能带
  `image_url`、`sha256`、`width`、`height`、`mime_type`、`metadata`。

当前构建中，`local_path` 通常是 cache image directory 内的绝对路径。
`relative_path` 的基准在不同 builder/历史版本间并不完全一致，不能假定它总是相对
dataset root。推荐的解析顺序是：

1. 使用存在且哈希匹配的 `local_path`；
2. 若数据经过正式打包，再尝试 `<dataset_root>/<relative_path>`；
3. 文件仍不存在时，将该图片记为 missing asset，而不是重新抓取网络资源。

EntiTables 的常见 `source` 是 `wikipedia_extract_chunk` 和
`wikipedia_image_download`；WDC 常见的是 `wdc_page_text_chunk`、
`wdc_image_column` 和 `wdc_page_image`。下游应根据 `asset_type` 处理模态，不应
根据 `source` 硬编码数据集分支。

### 8.2 `evidence_recoveries`

每条记录提供一条行级监督路径：

```text
query_table_id/query_row_id
  -> evidence.asset_id
  -> target_table_id/target_row_ids
```

关键字段为：

| 字段 | 含义 |
| --- | --- |
| `recovery_id`、`path_id` | recovery 和路径主键 |
| `query_table_id`、`query_row_id` | 路径起点 |
| `target_table_id`、`target_row_ids` | 路径终点 |
| `source_table_id`、`source_row_id` | 对应原表位置 |
| `query_entity` | query row 中的 entity 及可审计上下文 |
| `recovered_attribute` | 属性名、期望值、模型值及隐藏状态 |
| `evidence.asset_id` | 引用的 `bridge_assets.asset_id` |
| `evidence.asset_type` | `text` 或 `image` |
| `path_nodes` | query → asset → target 的显式节点序列 |

一个 query row 可以有零条、一条或多条 recovery。做 path-aware 训练时可以把每条
recovery 当作正路径，但应按 `query_table_id`、`query_row_id` 和 `asset_id` 去重。
`model_evidence`、`model_connection_evidence` 是解释信息，不是独立 ground truth。

### 8.3 其他关联 artifact

- `entities`：以 `entity_id` 为主键；WDC 可能额外包含 `page_url` 和
  `image_urls`。
- `table_asset_links`：连接 `source_table_id + row_id + column_index`、
  `entity_id` 和 `asset_ids`。
- `attribute_extractions`：记录单次 asset/entity 推理的
  `candidate_attribute_names`、`attributes`、`error` 和 `cache_key`。其中
  `raw_response` 仅供调试，不应直接作为训练文本。

## 9. `splits.json` 与无泄漏划分

`splits.json` 的顶层键为 `train`、`dev`、`test`，每个 split 包含：

```json
{
  "source_table_ids": [],
  "query_table_ids": [],
  "data_lake_table_ids": []
}
```

推荐的数据选择方式：

| 阶段 | Query | Candidate corpus | Ground truth |
| --- | --- | --- | --- |
| 训练 | `split == train` 的 query | train data-lake IDs | train qrels |
| 验证 | `split == dev` 的 query | dev data-lake IDs | dev qrels |
| 测试 | `split == test` 的 query | test data-lake IDs | test qrels |

必须以 source table 为泄漏边界。同一个 `source_table_id` 的 query、target、row views
和 evidence 不得跨 split。对 train 中同一 `chain_id` 的多个 `row_view_index`，可以
作为数据增强使用，但验证/测试只应出现 canonical `row_view_index = 0`。

## 10. 推荐的训练样本构造

### 10.1 普通 table retrieval

一条训练样本可表示为：

```text
(query_table_id, positive_target_table_id, negative_target_table_ids, split)
```

构造步骤：

1. 以 `qrels.jsonl` 生成正例 pair。
2. 从相同 split 的 `data_lake_table_ids` 中采样负例。
3. 排除该 query 的所有 qrel target；同时建议排除同一 `source_table_id` 的其他投影，
   以减少伪负例。
4. query 与 candidate 都只序列化可见的列名和 cell `text`。

### 10.2 Multimodal path retrieval

可以从 `evidence_recoveries` 构造：

```text
(query table/row, positive asset, positive target table/row)
```

文本素材读取 `bridge_assets.content`；图片素材读取本地文件。若只训练普通
query-to-table retriever，不应把正例 evidence 或 recovered value 拼入 query，
否则会产生标签泄漏。

### 10.3 不应进入普通 encoder 输入的字段

至少排除：

- `role`、`split`、`chain_id`、`source_table_id`；
- `source_column_indices`、`source_row_indices`、`provenance`；
- `hidden_attributes`、`target_table_ids`、`join_attribute`；
- qrels、label、positive/negative 标记；
- `recovered_attribute.value` 和任何只在监督侧可见的 evidence path 注释。

这些字段可用于采样、分组、监督和误差分析，但不能作为普通检索编码器的内容特征。

## 11. 推荐评测协议

对 dev/test 的每个 query，在相同 split 的完整 candidate corpus 上排序，并至少报告：

- Recall@K；
- MRR；
- nDCG@K（如果未来引入多级 `rel`）；
- query 数量与 candidate 数量。

建议同时给出三个切片：

1. 全部 query；
2. implicit：`reason == "model_recoverable_join_column"`；
3. explicit：`reason == "explicit_visible_join_column"`。

同一 train chain 的 row views 不能在评测时被当成独立测试难例重复计权。当前 dev/test
只有 row view 0；若未来格式扩展，评测应按 `chain_id` 或 canonical view 去重。

## 12. EntiTables 与 WDC 的允许差异

下游必须依赖共同字段，而不是要求两套记录逐字段相等。

| 范围 | EntiTables | WDC |
| --- | --- | --- |
| query/target/qrel 构造 | 共同实现 | 共同实现 |
| table provenance builder | `build_mm_joinability_dataset.py` | `build_wdc_mm_joinability_dataset.py` |
| entity 来源 | Wikipedia title/link | WDC row + page URL |
| asset 来源 | Wikipedia extract/image | page text、直接 image 属性、page image |
| manifest 扩展 | source sampling、Wikipedia cache/media | source corpus、web cache、安全限制、pipeline provenance |
| 额外诊断 | Wikipedia/API 统计 | web/media/model failure JSONL |

因此，schema 校验应采用“必需共同字段 + 允许来源扩展字段”的方式，不应使用严格的
整条 object key 相等判断。

## 13. 发布前校验清单

- manifest 列出的所有 shard 都存在，记录数与 manifest 相符；
- `table_id`、`source_table_id`、`asset_id`、`recovery_id` 在各自主表中唯一；
- 每条 qrel 的 query 和 target 均存在，且三者 split 一致；
- 每个 implicit query 恰好有一个 qrel；
- qrel 的 target 出现在对应 split 的 `data_lake_table_ids` 中；
- raw data-lake 引用能够解析到同 ID 的 source table；
- query 的 source rows 是同 chain target source rows 的子集；
- implicit join attribute 不出现在 query 可见列中；
- evidence 引用的 query、target 和 asset 均存在；
- train/dev/test 之间没有重复 `source_table_id`；
- 普通 encoder 序列化结果不包含隐藏属性、target ID 或 qrel 信息。
