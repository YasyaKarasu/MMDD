# EntiTables Multimodal Table Dataset 构建说明

本文档总结当前脚本 `scripts/build_mm_table_dataset.py` 的数据集构建逻辑，以及本次已生成的 `output_medium` 数据集版本的组成。

当前构建目标是：从 EntiTables JSON 表格中构造一个 multimodal table data lake，并生成 query workload。这个流程不是 joinability benchmark label 构造流程，不生成 positive/negative pairs，不生成 joinable/not-joinable labels，也不把 hidden columns 当作 ground truth。

## 本次构建命令

```powershell
python scripts\build_mm_table_dataset.py `
  --input_dir dataset\tables_redi2_1 `
  --output_dir output_medium `
  --max_tables 20000 `
  --max_query_views_per_source_table 5 `
  --max_entities 40000 `
  --max_images_per_entity 2 `
  --wiki_link_threshold 0.3 `
  --min_rows 6 `
  --min_cols 4 `
  --sleep 0.2 `
  --seed 13 `
  --split_by page_title `
  --flush_every_records 500 `
  --records_per_shard 50000
```

## 核心概念

### Source Table

Source table 是从 EntiTables 原始 JSON 中清洗、规范化后保留下来的原始表格。脚本会遍历输入目录下的 `.json` 文件，每个 JSON 文件是一个 table id 到 table object 的字典。

每张 source table 会包含：

- `source_table_id`
- `source_file`
- `page_title`
- `caption`
- `section_title`
- `columns`
- `rows`
- `metadata.column_profiles`
- `metadata.candidate_entity_columns`

其中 `candidate_entity_columns` 只表示候选实体列，不表示 joinability label。

### Query View

Query view 是从 source table 投影得到的查询表，是 query workload 的一部分。一个 source table 可以生成多个 query views。

query view 不是正样本，不是负样本，也不是 query-target pair。它只是后续系统实验中可用的查询表。

当前支持的 query view 构造策略：

- `entity_plus_one_attr`：选择一个候选实体列，加一个属性列。
- `entity_plus_two_attrs`：选择一个候选实体列，加两个属性列。
- `entity_only`：只保留候选实体列。
- `random_projection`：随机选择 2 到 4 列，用于增加 query workload 多样性。

每张 source table 最多生成 `--max_query_views_per_source_table` 个 query views。本次设置为 5。

### Entity

Entity 是从表格 cell 中解析出来的 Wikipedia entity。比如：

```text
[Ai_Takahashi|Ai Takahashi]
```

会被解析为：

```json
{
  "text": "Ai Takahashi",
  "wiki_title": "Ai Takahashi"
}
```

同一个 `wiki_title` 在 `entities/part-*.jsonl` 中只保留一个 entity，并记录其在 source tables 和 query views 中出现的位置。

注意：`--max_entities` 不限制实体抽取总量。它只限制最多对多少个 entity 调用 Wikipedia API、抓取文本和图片资产。本次 `--max_entities 40000` 表示最多为前 40000 个实体抓取 Wikipedia assets。

### Bridge Asset

Bridge asset 是与 entity 关联的多模态资产。当前包括：

- Wikipedia 页面文本摘要，即 `asset_type = "text"`。
- Wikipedia / Commons 图片，即 `asset_type = "image"`。

图片会被下载到本地 `output_medium/images/`，同时在 `bridge_assets/part-*.jsonl` 中保留原始 URL、本地路径、文件大小、SHA256 和 MediaWiki `imageinfo` metadata。

### Table Asset Link

Table asset link 记录某个 source table 或 query view 中的实体 cell 与 entity assets 的关联。

如果某个 entity 没有抓到 assets，也会保留 link，只是 `asset_ids` 为空列表。

这些 links 不是 labels，也不是正负样本。

## 表格解析与清洗逻辑

脚本会跳过以下表格：

- 空表。
- `data` 为空。
- `numDataRows == 0`。
- 列数小于 `--min_cols`。
- 行数小于 `--min_rows`。
- 行列数严重不一致。
- 有效非空单元格比例过低。

本次构建参数为：

- `--min_rows 6`
- `--min_cols 4`

列名规范化规则：

- 空列名改为 `col_0`、`col_1` 等。
- 重复列名追加后缀，例如 `Name`、`Name_1`。
- 原始列名保留在 `raw_column_name`。

单元格 Wikipedia link 解析支持：

- `[Page_Title|Display Text]`
- `[Page_Title]`
- 普通纯文本

`Category:`、`File:`、`Image:` 等非实体命名空间链接会被识别为 link，但默认不会作为 entity。

## 候选实体列识别逻辑

每个 source table 的每一列会计算 column profile：

- `wiki_link_ratio`
- `non_empty_ratio`
- `unique_ratio`
- `numeric_ratio`
- `avg_text_length`
- `is_candidate_entity_column`

候选实体列主要依据：

- `wiki_link_ratio >= --wiki_link_threshold`
- 非空比例足够高。
- 数值比例不高。
- unique ratio 不过低。
- 过滤明显不是实体的列，例如年份、序号、分数等。

本次构建参数为：

- `--wiki_link_threshold 0.3`

## Wikipedia 多模态资产逻辑

在未使用 `--no_wikipedia` 时，脚本会对最多 `--max_entities` 个实体调用 MediaWiki Action API。

本次设置：

- `--max_entities 40000`
- `--max_images_per_entity 2`
- `--sleep 0.2`

对每个被抓取的 entity：

1. 获取 Wikipedia 页面 extract。
2. 获取 page image / thumbnail / embedded images。
3. 对候选图片调用 `imageinfo`。
4. 过滤明显低价值图片，例如 icon、logo、flag、placeholder、小尺寸图标等。
5. 下载通过过滤的图片到 `output_medium/images/`。
6. 在 `bridge_assets/part-*.jsonl` 中记录文本资产和图片资产。

脚本不会因为单个 API 失败或单张图片下载失败而中断，而是记录 warning，并继续处理后续实体。

## Sharded JSONL 输出逻辑

为避免单个 JSONL 文件过大，当前输出采用分片目录结构。

每类大型 artifact 是一个目录，目录下是多个 part 文件：

```text
output_medium/
  source_tables/
    part-00000.jsonl
  query_views/
    part-00000.jsonl
  entities/
    part-00000.jsonl
    part-00001.jsonl
  bridge_assets/
    part-00000.jsonl
    part-00001.jsonl
  table_asset_links/
    part-00000.jsonl
    ...
  images/
  cache/
  dataset_manifest.json
  splits.json
  stats.json
```

本次设置：

- `--records_per_shard 50000`
- `--flush_every_records 500`

`dataset_manifest.json` 记录了每个 artifact 的目录、shard 文件和每个 shard 的记录数。后续读取数据时，建议以 `dataset_manifest.json` 为准，而不是直接 glob 目录，避免复用输出目录时误读旧文件。

## 当前 `output_medium` 数据集组成

根据 `output_medium/stats.json`，本次构建结果如下：

| 指标 | 数量 |
|---|---:|
| processed raw tables | 20000 |
| skipped raw tables | 13507 |
| source tables | 6493 |
| query views | 24795 |
| unique wiki entities | 91032 |
| text assets | 39480 |
| image assets | 32401 |
| table asset links | 765854 |
| API failures | 349 |
| avg query views per source table | 3.818728 |
| avg rows per query view | 23.833636 |
| avg columns per query view | 2.093406 |

跳过表格原因：

| skipped reason | count |
|---|---:|
| too_few_columns | 8393 |
| too_few_rows | 4127 |
| empty_data | 986 |
| low_non_empty_cell_ratio | 1 |

需要注意：

- `unique_wiki_entities = 91032`，这是从所有保留 source tables 和 query views 中抽取到的唯一实体数。
- `--max_entities 40000` 只限制 Wikipedia asset 抓取数量，因此最终 entity 总数可以大于 40000。
- `text_assets + image_assets = 71881`，与 `bridge_assets` 总记录数一致。
- `image_assets = 32401`，对应本地下载图片数量。

## 当前 shard 布局

根据 `output_medium/dataset_manifest.json`：

| artifact | records | shards |
|---|---:|---:|
| source_tables | 6493 | 1 |
| query_views | 24795 | 1 |
| entities | 91032 | 2 |
| bridge_assets | 71881 | 2 |
| table_asset_links | 765854 | 16 |

具体 shard：

- `source_tables/part-00000.jsonl`: 6493 records
- `query_views/part-00000.jsonl`: 24795 records
- `entities/part-00000.jsonl`: 50000 records
- `entities/part-00001.jsonl`: 41032 records
- `bridge_assets/part-00000.jsonl`: 50000 records
- `bridge_assets/part-00001.jsonl`: 21881 records
- `table_asset_links/part-00000.jsonl` 到 `part-00014.jsonl`: 每个 50000 records
- `table_asset_links/part-00015.jsonl`: 15854 records

## 当前 split 组成

本次使用：

```text
--split_by page_title
```

split 是 source-level split，不是 pair-level split。也就是说，同一个 source table 的所有 query views 都会进入同一个 split；同一个 page title 下的 source tables 也会尽量进入同一个 split，从而降低同源泄漏风险。

当前 split 数量：

| split | source tables | query views |
|---|---:|---:|
| train | 5280 | 20250 |
| dev | 610 | 2235 |
| test | 603 | 2310 |

`splits.json` 中的说明为：

```text
splits are source-level to avoid leakage across query views
```

## 图片与缓存

当前图片目录：

```text
output_medium/images/
```

图片统计：

- downloaded image files: 32401
- total image bytes: 39738255434
- approximate total size: 37.01 GiB

缓存目录：

```text
output_medium/cache/
```

缓存文件：

| cache file | size |
|---|---:|
| `wiki_pages.jsonl` | 425942894 bytes |
| `wiki_images.jsonl` | 242736032 bytes |

缓存用于避免重复调用 MediaWiki API。

## 明确不包含的内容

当前数据集构建流程不会生成以下内容：

- `original_joinable`
- `bridged_joinable`
- `negative_type`
- `positive_pairs`
- `negative_pairs`
- `labels`
- joinability labels
- positive / negative examples
- query-target pairs
- augmentation ground truth

尤其需要注意：

- `query_views` 是 query workload，不是训练标签。
- `hidden_column_indices` 和 `hidden_column_names` 只是投影 provenance，不是 target，也不是 ground truth。
- `table_asset_links` 只是实体 cell 到多模态 assets 的连接，不是 positive/negative evidence label。

## 后续读取建议

推荐读取顺序：

1. 读取 `dataset_manifest.json`。
2. 按 manifest 中列出的 shards 读取各 artifact。
3. 使用 `source_table_id` 和 `query_view_id` 关联 source tables、query views、splits 和 table asset links。
4. 使用 `entity_id` 关联 `entities`、`bridge_assets` 和 `table_asset_links`。

不要假设某个 artifact 一定只有一个 JSONL 文件；应始终按 manifest 读取。
