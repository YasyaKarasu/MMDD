# EntiTables 多模态表格数据集构建脚本

本项目用于从 EntiTables JSON 文件中构造一个 multimodal table data lake，并进一步生成 query workload。

这个脚本只负责构造数据集本体和 query views，不负责构造 joinability benchmark labels。它不会生成正样本、负样本、joinable/not joinable 标签、augmentation target 或 query-target pairs。

## 输入格式

每个 EntiTables JSON 文件应是一个顶层字典：

```json
{
  "table-0001-590": {
    "title": ["Column A", "Column B"],
    "numCols": 2,
    "numericColumns": [],
    "pgTitle": "Wikipedia page title",
    "numDataRows": 4,
    "secondTitle": "Section title",
    "numHeaderRows": 1,
    "caption": "Table caption",
    "data": [
      ["[Page_Title|Display Text]", "plain text"],
      ["[Another_Page]", "123"]
    ]
  }
}
```

单元格支持以下形式：

- `[Page_Title|Display Text]`
- `[Page_Title]`
- 普通纯文本

脚本会解析 Wikipedia link，清理下划线、HTML entity 和前后空白。`Category:`、`File:`、`Image:` 等非实体命名空间链接会被识别为链接，但默认不会当作实体。

## 输出文件

脚本会在 `--output_dir` 下生成：

- `source_tables.jsonl`：清洗后的标准化 source tables，包括列信息、行单元格、列画像和候选实体列。
- `query_views.jsonl`：从 source table 投影得到的 query workload tables。
- `entities.jsonl`：从 source tables 和 query views 中抽取到的唯一 Wikipedia entities。
- `bridge_assets.jsonl`：可选的 Wikipedia 文本摘要，以及已下载图片的本地路径、原始 URL 和 metadata。
- `table_asset_links.jsonl`：实体 cell 与 multimodal assets 的关联。
- `splits.json`：source-level 的 train/dev/test 划分。
- `stats.json`：处理数量、跳过原因、资产数量、query workload 统计等。

## Source Tables

`source_tables.jsonl` 中每一行是一张标准化后的原始表。

脚本会跳过以下表格：

- 空表
- `data` 为空
- `numDataRows == 0`
- 列数小于 `--min_cols`
- 行数小于 `--min_rows`
- 行列数严重不一致
- 有效非空单元格比例过低

列名会被规范化：

- 空列名改为 `col_0`、`col_1` 等。
- 重复列名追加后缀，例如 `Name`、`Name_1`。
- 原始列名保存在 `raw_column_name` 中。

每一列都会生成 column profile：

- `wiki_link_ratio`
- `non_empty_ratio`
- `unique_ratio`
- `numeric_ratio`
- `avg_text_length`
- `is_candidate_entity_column`

候选实体列只表示“这列可能是实体列”，不表示任何 joinability label。

## Query Views

`query_views.jsonl` 是查询工作负载的一部分。一个 source table 可以生成多个 query views。

query view 只表示从 source table 投影得到的查询表，不表示 positive，也不表示 negative。

支持的生成策略：

- `entity_plus_one_attr`：一个候选实体列加一个属性列。
- `entity_plus_two_attrs`：一个候选实体列加两个属性列。
- `entity_only`：只保留候选实体列。
- `random_projection`：随机选择 2 到 4 列，用于增加 workload 多样性。

每张 source table 最多生成 `--max_query_views_per_source_table` 个 query views。采样由 `--seed` 控制，可复现。

## Hidden Columns

`hidden_column_indices` 和 `hidden_column_names` 只表示构造 query view 时被隐藏的原始列。

它们只是 provenance 信息，不是：

- augmentation ground truth
- target columns
- positive labels
- negative labels
- joinability labels
- query-target pairs

后续实验可以知道 query view 来源于哪张 source table，但不能把 hidden columns 直接当作训练标签。

## Entities

`entities.jsonl` 从 source tables 和 query views 中出现的 `wiki_title` 抽取实体。

同一个 `wiki_title` 只保留一个 entity，并记录它出现在哪些 cell 中：

- `source_table_id`
- `query_view_id`
- `row_id`
- `column_index`
- `column_name`

即使使用 `--no_wikipedia`，脚本也会生成 `entities.jsonl`。

## Wikipedia 多模态资产

默认情况下，脚本会用 MediaWiki Action API 抓取实体的 Wikipedia assets：

- 页面 extract
- page image / thumbnail
- embedded images
- imageinfo metadata
- Commons `extmetadata`

脚本会下载通过过滤的图片本体，并保留图片原始 URL 和 metadata。

图片默认下载到：

```text
<output_dir>/images/
```

每条 image asset 会包含：

- `image_url`：Wikipedia / Commons 原始图片 URL。
- `description_url`：图片说明页 URL。
- `local_path`：下载后的本地绝对路径。
- `relative_path`：相对于 `output_dir` 的路径。
- `file_name`：本地图片文件名。
- `bytes`：下载文件大小。
- `sha256`：本地文件哈希。
- `metadata`：MediaWiki `imageinfo` metadata，包括 `extmetadata`。

脚本仍会在下载前过滤明显无用的图片，例如 icon、logo、flag、placeholder 等。

本地缓存写入：

- `cache/wiki_pages.jsonl`
- `cache/wiki_images.jsonl`

如果 API 失败，脚本会记录 warning，并继续处理后续实体。

## Table Asset Links

`table_asset_links.jsonl` 记录 source table / query view 中实体 cell 与 assets 的关联。

即使没有抓到 Wikipedia assets，也会保留 entity link，只是 `asset_ids` 为空列表。

这些 links 不是正负样本标签，也不是 joinability labels。

## 数据划分

`splits.json` 是 source-level split，不是 pair-level split。

可用参数：

- `--split_by source_table_id`
- `--split_by page_title`

默认推荐使用：

```powershell
--split_by page_title
```

这样同一个 Wikipedia 页面下的 source tables 会尽量进入同一个 split，避免页面级泄漏。

无论使用哪种 split key，同一张 source table 的所有 query views 都会跟随该 source table 进入同一个 split。

默认比例：

- train: 0.8
- dev: 0.1
- test: 0.1

## 环境配置

本机已创建 conda 环境：

```powershell
conda activate mm-table-dataset
```

环境中包含：

- Python 3.10
- requests
- tqdm

如果需要重新创建环境：

```powershell
conda create -n mm-table-dataset python=3.10 requests tqdm -y
conda activate mm-table-dataset
```

## 使用方式

构造表格数据集、query workload，并抓取 Wikipedia metadata：

```powershell
python scripts/build_mm_table_dataset.py `
  --input_dir dataset\tables_redi2_1 `
  --output_dir output `
  --max_tables 100 `
  --max_query_views_per_source_table 5 `
  --max_entities 1000 `
  --max_images_per_entity 3 `
  --wiki_link_threshold 0.3 `
  --split_by page_title
```

只构造表格、query views、entities 和 table asset links，不调用 Wikipedia：

```powershell
python scripts/build_mm_table_dataset.py `
  --input_dir dataset\tables_redi2_1 `
  --output_dir output_no_wikipedia `
  --max_tables 100 `
  --max_query_views_per_source_table 5 `
  --wiki_link_threshold 0.3 `
  --split_by page_title `
  --no_wikipedia
```

使用 `--no_wikipedia` 时：

- `bridge_assets.jsonl` 可以为空。
- `entities.jsonl` 仍会生成。
- `table_asset_links.jsonl` 仍会生成。
- 每条 link 的 `asset_ids` 可以是空列表。

## 常用参数

- `--input_dir`：EntiTables JSON 输入目录。
- `--output_dir`：输出目录。
- `--max_tables`：最多处理多少张原始表。
- `--max_query_views_per_source_table`：每张 source table 最多生成多少个 query views。
- `--max_entities`：最多为多少个实体抓取 Wikipedia assets。
- `--max_images_per_entity`：每个实体最多下载并保留多少张图片。
- `--wiki_link_threshold`：候选实体列的 Wikipedia link 比例阈值。
- `--min_rows`：最少数据行数。
- `--min_cols`：最少列数。
- `--sleep`：MediaWiki API 请求间隔。
- `--seed`：随机种子。
- `--no_wikipedia`：不调用 Wikipedia API。
- `--split_by`：`source_table_id` 或 `page_title`。
- `--train_ratio`：训练集比例。
- `--dev_ratio`：开发集比例。
- `--test_ratio`：测试集比例。

## 明确不生成的内容

本脚本不会生成：

- `original_joinable`
- `bridged_joinable`
- `negative_type`
- `positive_pairs`
- `negative_pairs`
- `labels`
- positive / negative examples
- joinable / not joinable labels
- query-target pairs
- augmentation ground truth

本脚本的目标是构造 multimodal table dataset + query workload，而不是构造 joinability benchmark labels。

## 流式写入

为了降低完整 EntiTables 构建时的内存和单文件压力，脚本现在会对大体量 JSONL artifact 进行增量写入、定期 flush，并按记录数切分成多个 shard 文件，而不是等整个数据集全部处理完后再一次性写成一个巨大 JSONL。

输出结构现在是每类 artifact 一个目录，每个目录下放多个 part 文件：

- `source_tables/part-*.jsonl`
- `query_views/part-*.jsonl`
- `entities/part-*.jsonl`
- `bridge_assets/part-*.jsonl`
- `table_asset_links/part-*.jsonl`

脚本还会生成：

- `dataset_manifest.json`：记录每个 artifact 的目录、part 文件列表和记录数。
- `splits.json`：source-level split。
- `stats.json`：统计信息。

可以通过 `--flush_every_records` 控制 flush 频率，默认值是 `500`。

可以通过 `--records_per_shard` 控制每个 part 文件最多包含多少条 JSONL 记录，默认值是 `50000`：

```powershell
python scripts/build_mm_table_dataset.py `
  --input_dir dataset\tables_redi2_1 `
  --output_dir output `
  --max_query_views_per_source_table 5 `
  --max_entities 1000 `
  --max_images_per_entity 3 `
  --wiki_link_threshold 0.3 `
  --flush_every_records 500 `
  --records_per_shard 50000 `
  --split_by page_title
```

后续读取数据时建议优先读取 `dataset_manifest.json` 中列出的 shards，而不是直接 glob 目录。这样即使复用旧的 `output_dir`，也不会误读旧 run 留下的 stale part 文件。

## Stage-1 逻辑连通性 Pipeline

Stage-1 会基于已有的 `output_medium` artifact 构造 coarse recall benchmark 和训练流程。它不会重新抓取 Wikipedia，不会重建 `output_medium`，也不会把 `query_views` 当成监督标签。

推荐使用下面几个整合入口：

```bash
# 1) 生成 Stage-1 初始数据：logic fragments、evidence paths、weak labels、冻结 embedding。
conda run -n MMDD python scripts/stage1_prepare_data.py \
  --input_dir output_medium \
  --stage1_dir output_stage1_logic \
  --encoder_path ./Qwen3-VL-Embedding-2B \
  --embedding_batch_size 8 \
  --device cuda \
  --dtype bf16

# 2) 训练 teacher，运行 HITL 标注轮次，重新训练 teacher，并蒸馏 student。
conda run -n MMDD python scripts/stage1_train_models.py \
  --stage1_dir output_stage1_logic \
  --hitl_rounds 3 \
  --hitl_batch_size 50 \
  --distill_loss pairwise \
  --force_retrain \
  --lan \
  --port 7860

# 3) 构建 ANN 索引，并评估 relation-aware multi-hop recall。
conda run -n MMDD python scripts/stage1_index_eval.py \
  --stage1_dir output_stage1_logic \
  --topk 10 50 100 \
  --max_hops 3 \
  --beam_width 64 \
  --beam_neighbors 50
```

如果希望一个命令跑完整一阶段，可以使用 `stage1_run_all.py`。它会在 HITL 轮次中启动浏览器标注 GUI，并等待人工标签 merge 完成后继续：

```bash
conda run -n MMDD python scripts/stage1_run_all.py \
  --input_dir output_medium \
  --stage1_dir output_stage1_logic \
  --hitl_rounds 3 \
  --hitl_batch_size 50 \
  --distill_loss pairwise \
  --force_retrain \
  --lan \
  --port 7860
```

如果想在浏览器里查看已经构造出来的连接组，可以运行：

```bash
conda run -n MMDD python scripts/stage1_connection_viewer.py --stage1_dir output_stage1_logic --port 7861
```

这个 viewer 会按 `chain_id` 分页展示，每页包含 visible query、hidden query、target fragment、logic pairs、qrels 和 evidence paths。

默认情况下，Flask GUI 绑定 `127.0.0.1`，只能在本机访问。给 `stage1_train_models.py`、`stage1_run_all.py`、`run_hitl_training_rounds.py`、`hitl_annotation_app.py` 或 `stage1_connection_viewer.py` 加上 `--lan` 后会绑定到 `0.0.0.0`，启动日志会同时打印本机 URL 和内网设备可访问的 URL。如果内网 URL 仍然打不开，检查主机防火墙是否放行对应端口，并确认设备在同一个网络里。

如果想忽略之前跑到一半留下的训练输出，并让 training/HITL loop 干净重跑，加上 `--force_retrain`。它会删除 teacher/student 模型、teacher scores、train pairs、未完成的 HITL 选择、标注模板和标注状态文件，但保留准备阶段数据、embeddings 和已经 merge 的人工标签。只有在你也想丢弃已合并人工标签时，才额外加 `--reset_human_labels`。

下面这些底层命令仍然可以用于调试或单独运行某一步：

```bash
conda run -n MMDD python scripts/build_stage1_logic_connectivity.py --input_dir output_medium --output_dir output_stage1_logic --seed 13
conda run -n MMDD python scripts/build_stage1_evidence_paths.py --input_dir output_medium --stage1_dir output_stage1_logic --seed 13
conda run -n MMDD python scripts/generate_weak_labels.py --stage1_dir output_stage1_logic
conda run -n MMDD python scripts/select_hitl_batch.py --stage1_dir output_stage1_logic --round_id 0 --batch_size 50
conda run -n MMDD python scripts/build_stage1_embeddings.py --input_dir output_medium --stage1_dir output_stage1_logic --encoder_path ./Qwen3-VL-Embedding-2B --batch_size 8 --device cuda --dtype bf16
conda run -n MMDD python scripts/build_teacher_training_data.py --stage1_dir output_stage1_logic
conda run -n MMDD python scripts/train_teacher.py --stage1_dir output_stage1_logic --embedding_dir output_stage1_logic/embeddings
conda run -n MMDD python scripts/train_student.py --stage1_dir output_stage1_logic --embedding_dir output_stage1_logic/embeddings --teacher_scores output_stage1_logic/teacher_scores.jsonl --distill_loss pairwise
conda run -n MMDD python scripts/build_hnsw_indices.py --stage1_dir output_stage1_logic --student_dir output_stage1_logic/student --space cosine --m 32 --ef_construction 200 --ef_search 100
conda run -n MMDD python scripts/eval_stage1_recall.py --stage1_dir output_stage1_logic --student_dir output_stage1_logic/student --hnsw_dir output_stage1_logic/hnsw_indices --topk 10 50 100 --max_hops 3 --beam_width 64 --beam_neighbors 50
```

`train_student.py` 支持 `--distill_loss pairwise` 和 `--distill_loss listwise`。ANN index 中存的是被 student 投影后的对象向量 `u_b`；在线检索时会先计算关系感知 query vector `u_a R_{\tau(a),\tau(b)}`，再去查对应 target type 的索引。path-aware evaluation 使用 beam search 做 multi-hop 扩展，最终只返回 table endpoints，assets 只作为路径中的中间证据保留。

填完 `output_stage1_logic/human_labels_template_round_0.jsonl` 后，可以用下面的命令合并人工标签：

```bash
conda run -n MMDD python scripts/merge_human_labels.py --stage1_dir output_stage1_logic --human_labels output_stage1_logic/human_labels_template_round_0.filled.jsonl
```

如果希望运行主动学习闭环，让 teacher 每轮重新训练、对候选路径打分、挑选不确定样本，并通过浏览器 GUI 收集人工标签：

```bash
conda run -n MMDD python scripts/run_hitl_training_rounds.py \
  --stage1_dir output_stage1_logic \
  --embedding_dir output_stage1_logic/embeddings \
  --rounds 3 \
  --batch_size 50 \
  --force_retrain \
  --lan \
  --port 7860
```

每一轮会执行：基于最新已合并人工标签构造训练数据，训练 teacher，写出 `teacher_scores.jsonl`，选择不确定的 evidence paths，生成 `human_labels_template_round_<N>.jsonl`，启动 Flask 标注网页，并等待网页把完整标注合并到 `human_labeled_paths.jsonl`。默认情况下，选样器会排除已经合并的人工标签样本，以及之前 round 已选但还没完成标注的 open batch。

如果只想标注已有 round，而不启动训练闭环：

```bash
conda run -n MMDD python scripts/hitl_annotation_app.py --stage1_dir output_stage1_logic --round_id 0 --lan --port 7860
```

然后在浏览器打开：

```text
http://<启动日志里显示的内网 IP>:7860
```

默认 encoder 是本地 `Qwen3-VL-Embedding-2B`。如果 `./Qwen3-VL-Embedding-2B` 不存在，wrapper 也会检查 `./hf_models/Qwen3-VL-Embedding-2B`；其他本地路径可以通过 `--encoder_path` 指定。
