# EntiTables 多模态表格数据集构建脚本

本项目用于从 EntiTables JSON 文件中构造一个 multimodal table data lake，并进一步生成 query workload。

EntiTables 与 WDC canonical joinability 输出的共同下游训练、负例采样和评测格式，
见 [`docs/mm_joinability_downstream_data_format.zh-CN.md`](docs/mm_joinability_downstream_data_format.zh-CN.md)。

当前可续跑、分片化的 WDC joinability 构建器位于 `src/`，使用说明见
[`src/README.md`](src/README.md)。新的 WDC 构建应使用
`src/build_wdc_dataset.py`；`scripts_old/wdc200k_*` 仅保留为旧行为参考。

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

## EntiTables 同时使用本地与远程 GPU 推理

`build_mm_joinability_dataset.py` 现在可以同时使用本地和远程的
OpenAI-compatible vLLM endpoint。两组请求使用完全独立的并发上限；同一模态的本地
worker 与远程 worker 从同一个待处理任务队列取任务，因此性能更好的远程显卡会自然
处理更多任务，但不会抬高本地显卡的并发。

原有 endpoint 和 worker 参数继续表示本地池，远程池使用单独参数：

```bash
export MMDD_REMOTE_TEXT_MODEL_API_KEY="..."   # 可选
export MMDD_REMOTE_IMAGE_MODEL_API_KEY="..."  # 可选
conda run --no-capture-output -n MMDD python \
  scripts/build_mm_joinability_dataset.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_mm_joinability \
  --text_model_base_url http://127.0.0.1:8001/v1 \
  --image_model_base_url http://127.0.0.1:8000/v1 \
  --text_model_workers 2 \
  --image_model_workers 1 \
  --remote_text_model_base_url http://127.0.0.1:18001/v1 \
  --remote_image_model_base_url http://127.0.0.1:18000/v1 \
  --remote_text_model_workers 12 \
  --remote_image_model_workers 8
```

上例假设通过 SSH tunnel 把远程 text/image 服务映射到本机的 18001/18000 端口。
远程服务暴露的 served model name 必须与 `--text_model_name`、
`--image_model_name` 一致。多个远程 endpoint 可通过
`--remote_text_model_base_urls` / `--remote_image_model_base_urls` 指定；动态列表可用
对应的 `*_base_urls_file`。某一远程 worker 数为 `0` 时，该远程模态关闭。没有提供
专用远程 key 时，会依次回退到 `VLLM_API_KEY` 和对应的本地模态 key。

### 显式可见 join query

joinability builder 可以把多模态恢复失败、但仍有普通可见 join 列的 source table
转成显式 join query。默认 `ratio` 模式使用固定 seed 的 0.2 比例，并保持原来的每张
source table 最多一个显式 query。若希望 train/dev/test 各自的显式 query 数量与
implicit query 数量配平，使用：

```bash
--explicit_join_fallback_mode match_implicit
```

`match_implicit` 是 query-level 配平：同一张 source table 可以按多个可行的可见 join
列贡献多个 explicit query。所有 sibling variant 的 join 列都只放在各自的 query/target
pair 中；普通 context 列则在整张 source table 上一次性划分为 query-only 和 target-only，
因此一个 sibling query 不会通过 context 列与另一个 sibling target 产生 joinability。

`stats.json` 会在 `explicit_join_candidate_*` 下分别记录候选 source-table 数量和候选
query 数量。

### WDC 优先的远端 GPU 反向借用

当远端 vLLM 由 layout control agent 管理时，EntiTables 可以在 WDC 不运行模型任务
期间借用同一组远端 GPU。本地 EntiTables worker 不会等待远端 lease；WDC 忙时它们
继续处理共享队列，WDC 释放 lease 后远端 worker 会自动加入。WDC 再次进入模型阶段
时，EntiTables 会先撤销远端路由、等待在途请求结束、释放 lease，然后确认 WDC 的
优先请求。

两边必须使用同一个 `--remote_layout_coordination_dir`。WDC 命令增加：

```bash
--remote_layout_control_url http://127.0.0.1:18999 \
--remote_layout_control_token_file layout-control-token \
--remote_layout_primary_image_url http://127.0.0.1:18000/v1 \
--remote_layout_switchable_url http://127.0.0.1:18001/v1 \
--remote_layout_coordination_dir work_gpu_priority/remote
```

EntiTables 使用动态本地 GPU runner，并增加相同的远端控制参数：

```bash
conda run --no-capture-output -n MMDD python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_mm_joinability \
  --text_model_path hf_models/Qwen3.5-9B \
  --image_model_path hf_models/Qwen3-VL-8B-Instruct \
  --gpu_coordination_dir work_gpu_priority/local \
  --remote_layout_control_url http://127.0.0.1:18999 \
  --remote_layout_control_token_file layout-control-token \
  --remote_layout_primary_image_url http://127.0.0.1:18000/v1 \
  --remote_layout_switchable_url http://127.0.0.1:18001/v1 \
  --remote_layout_coordination_dir work_gpu_priority/remote \
  --remote_text_model_workers 32 \
  --remote_image_model_workers 64
```

远端文本和图像并发是彼此独立的总数。本例中 switchable endpoint 提供 32 个文本
并发槽位；`image_burst` 布局下两个图像 endpoint 各提供 32 个槽位，因此图像 worker
总数为 64。

control token 文件必须是 `0600`，并且不能使用受保护的 `.env.openai`。远端推理 key
仍通过 `MMDD_REMOTE_TEXT_MODEL_API_KEY`、`MMDD_REMOTE_IMAGE_MODEL_API_KEY` 或
`VLLM_API_KEY` 提供。若 WDC 异常退出且留下 `priority_requested`，EntiTables 会保持
远端路由关闭，直到新的 WDC 运行发布 `borrowable`，这是有意的 fail-closed 行为。

## 使用 OpenAI 构造 EntiTables Joinability 数据集

`build_mm_joinability_dataset_openai.py` 使用同一个非流式 OpenAI Chat Completions
模型完成文本和图片的属性抽取，其余部分完全复用现有
`build_mm_joinability_dataset.py` 的抽取 prompt、解析器、采样、恢复判定、缓存和
canonical 输出格式。

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
conda run --no-capture-output -n MMDD python \
  scripts/build_mm_joinability_dataset_openai.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_mm_joinability_openai \
  --cache_dir cache/mm_joinability \
  --openai_model gpt-5.6 \
  --openai_reasoning_effort none \
  --text_model_workers 4 \
  --image_model_workers 4 \
  --openai_max_inflight 4
```

也可以不在交互式 shell 中直接 export 密钥，而是创建一个已被本仓库
`.gitignore` 排除的 dotenv 文件：

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

把文件权限设为 `0600`。两个 OpenAI 构造脚本会在 `./.env.openai` 存在时自动读取；
如需使用其他文件，可传入 `--openai_env_file /path/to/another.env`。加载器只接受上述
两个变量，不会把文件当作 shell 执行。文件中的值会覆盖 shell 遗留的旧值；显式
传入的 `--openai_base_url` 仍具有最高优先级。

API key 只从 `--openai_api_key_env` 指定的环境变量读取，默认变量名是
`OPENAI_API_KEY`，不会写入任何运行元数据。API base URL 会读取
`OPENAI_BASE_URL`，并可通过 `--openai_base_url` 覆盖。Wikipedia 下载和共享的模型抽取缓存
仍放在 `--cache_dir`。模型、reasoning effort、输出 token 上限、图片 detail 和
图片像素预算等所有会改变推理结果的配置都会进入 provider model identity，因此
不同配置不会误用彼此的抽取缓存。默认情况下，带指纹的运行配置和累计 token 用量
写入 `<output 父目录>/work_mm_joinability_openai/openai_model_runs/`；可通过
`--openai_work_root` 修改该根目录。

`--text_model_workers` 和 `--image_model_workers` 继续控制两个模态各自的任务池；
`--openai_max_inflight`（默认 `4`）是两个任务池共享的请求上限。可选的
`--openai_requests_per_minute` 与 `--openai_tokens_per_minute` 会主动平滑共享请求流，
值为 `0` 时关闭对应限制。临时错误采用指数退避，遵守服务端 `Retry-After`，并让两个
模态共享冷却窗口。EntiTables 在临时错误耗尽重试后会停止，但保留已经成功的缓存；
重新运行即可恢复，不会把限流造成的空结果当成负样本。

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

## WDC Schema.org 200K Joinability Pipeline

`build_wdc200k_mm_joinability_dataset.py` 是面向 20 万表规模的分阶段、可恢复入口。
`build_wdc200k_mm_joinability_dataset_openai.py` 在同一分阶段流水线上使用 OpenAI
推理，并支持上文的共享并发与速率限制参数；耗尽重试的临时模型任务会保持为
`retryable`，重新运行后继续处理。
`build_wdc_mm_joinability_dataset.py` 仍是旧的有界 builder；它的逐表行数上限与
内存模型不适用于正式 200K 运行。

三类根目录必须彼此独立：

```text
output_wdc_200k/   只放最终 canonical dataset artifacts
work_wdc_200k/     selection、shards、durable jobs、checkpoints、
                   stage manifests、runtime markers、progress.json
cache/wdc_webtable/    可复用的网络、媒体、模型成功及失败结果
```

不要让三者互相嵌套，也不要复用同一路径。只有 `output_dir` 包含 source/query/data
lake tables、entities、assets、links、extractions、qrels、splits、errors、
`stats.json` 与 `dataset_manifest.json`。source table 的行数没有上限，每个入选表
的所有行都会保留。不要传 `--max_rows_per_source_table`：兼容参数会拒绝任何值，
包括 0。WDC `image` 列只用于发现素材，不会进入表 artifact。

### Preflight、分阶段运行与恢复

下面的只读 preflight 只验证 statistics archives、路径隔离、runtime 路径、磁盘
reserve 与参数，不执行任何 stage：

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_webtable \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --dry_run
```

structural preflight 会在 `work_dir` 写确定性的 selection/structural 状态，但不会
执行网络与模型任务：

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_webtable \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --web_max_retries 0 \
  --web_max_response_seconds 8 \
  --web_global_concurrency 128 \
  --web_per_host_concurrency 2 \
  --max_image_attempts_per_entity 3 \
  --max_images_per_entity 3 \
  --stop_after structural
```

继续前先检查 `work_wdc_200k/progress.json` 与 structural manifests。用完全相同的
四个根目录和参数恢复；已完成且 checksum 有效的 shard、成功 URL 和 terminal
失败 URL 都不会重放：

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_webtable \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --resume \
  --text_model_base_url http://127.0.0.1:8001/v1 \
  --image_model_base_url http://127.0.0.1:8000/v1
```

`--stop_after` 用于建立 stage barrier。需要有意重建某个 stage 及其下游时使用
`--from_stage`；它会先验证上游，再归档被替换的状态，而不是删除：

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_webtable \
  --max_source_tables 200000 \
  --from_stage pages \
  --stop_after pages
```

`--refresh_page_cache` 只能配合 `--from_stage pages` 或更早 stage，
`--refresh_image_cache` 只能配合 `--from_stage images` 或更早 stage。普通
`--resume` 会同时复用成功和 terminal 失败。

### 真实表 100/1,000 规模门禁

完整 corpus 中 mandatory `top100` 候选数已经超过 100 和 1,000，因此不能仅把
完整输入上的 `--max_source_tables` 改成 100/1,000。应先创建隔离的、精确大小的
真实 WDC 表门禁输入：

```bash
conda run -n MMDD python scripts/create_wdc200k_scale_gate_input.py \
  --source_dir wdc_schemaorg_2023 \
  --target_dir gate_inputs/wdc_100 \
  --table_count 100 \
  --selection_mode global_lowest \
  --seed 13

conda run -n MMDD python scripts/create_wdc200k_scale_gate_input.py \
  --source_dir wdc_schemaorg_2023 \
  --target_dir gate_inputs/wdc_1000 \
  --table_count 1000 \
  --selection_mode global_lowest \
  --seed 13
```

这些 quick gate 按
`(rows, stable_hash(seed, relative_path), relative_path)` 在全局范围最小化源表
行数，适合验证 pipeline plumbing、中断/恢复、请求去重、checksum、资源边界和
canonical output。它们不代表 WDC 的 category balance、table quality 或 row-count
distribution，也不会改变正式的分层 200K 选择。helper 只生成 production parser
可读的过滤版 statistics ZIP 与指向绝对源文件的 gzip symlink；不会复制、删除、
抓取或修改 corpus。目标必须不存在或严格为空，解析后的 realpath 必须位于 corpus
树外（两者都不能包含对方）。`scale_gate_manifest.jsonl` 记录绝对 source、相对
target 与源 gzip SHA-256，`scale_gate_checksums.json` 覆盖 manifest 和过滤 ZIP。
所有内容先在同父唯一 staging 目录内构建并验证，再进行原子 no-clobber 发布；
malformed 或 duplicate catalog path 会 fail closed，不留下 partial target。这些
目录是运维规模门禁子语料，不是正式 200K 抽样结果。分别用它们作为 `--input_dir`，
配套 `--max_source_tables 100/1000` 和全新的 output/work/cache 根目录运行。10K
structural gate 直接使用完整 corpus。

### Direct endpoints、动态 vLLM 与 tmux

上面的 resume 命令就是 direct runner：operator 自行提供已启动的
OpenAI-compatible 文本/图片 endpoints。动态 runner 会启动并重分配两个 vLLM
模态，未占用参数会透传到 staged WDC 200K builder：

```bash
conda run --no-capture-output -n MMDD python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --text_model_path /path/to/text-model \
  --image_model_path /path/to/vision-model \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_webtable \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --resume
```

dynamic runner 独占 `work_wdc_200k/runtime` 下的 endpoint files、模型 identity 与
start/ready/done markers。

structural tmux preflight 应让进程 stdout 直接显示在 pane 中，并用第二个 window
读取原子 progress snapshot：

```bash
tmux new-session -d -s wdc_200k \
  'conda run --no-capture-output -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir wdc_schemaorg_2023 --output_dir output_wdc_200k --work_dir work_wdc_200k --cache_dir cache/wdc_webtable --max_source_tables 200000 --selection_seed 13 --stop_after structural'
tmux new-window -t wdc_200k -n progress \
  "watch -n 5 'conda run -n MMDD python -m json.tool work_wdc_200k/progress.json'"
tmux attach-session -t wdc_200k
```

不要把 stdout/stderr 重定向到 log 文件：runner 会直接输出有界的周期进度，持久
进度位于 `work_wdc_200k/progress.json`。规模门禁测量与验收字段记录在
`docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`。

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

如果想在浏览器里查看 `build_mm_joinability_dataset.py` 已生成的数据集，可以运行：

```bash
conda run -n MMDD python scripts/mm_joinability_dataset_viewer.py --output_dir output_mm_joinability --port 7863
```

这个 viewer 会按 `qrels.jsonl` 里的 `query_table_id -> target_table_id` 组合分页展示，每页包含 query table、target table、evidence recovery paths，以及路径上引用到的文本/图片素材。训练集里的不相交 row view 会按 `chain_id` 和 `row_view_index` 分组，可以通过 row-view 筛选器分别查看 canonical 或 augmented view。viewer 首次使用时会构建 `<output_dir>/.mm_joinability_viewer.sqlite3`，相关数据 shard 未变化时会直接复用，因此浏览 WDC 规模的数据集时不需要把所有 table 和 asset 常驻内存；可用 `--index_path` 把这个生成索引放到其他位置。

新数据集构建完成后，可以分别启动 EntiTables 和 WDC 的 1% implicit-query 质量抽检页面：

```bash
conda run -n MMDD python scripts/entitables_dataset_checker.py --lan
conda run -n MMDD python scripts/wdc_dataset_checker.py --lan
```

checker 使用固定 seed 精确抽取 1% 的唯一 implicit query，每页展示一条 `query -> multimodal evidence -> attribute -> target`。每个 query 可标记为“合格”或“不合格”并填写备注；审核状态保存在数据集目录内的 SQLite 文件中，重启后可以继续。全部样本审核完成后，页面会给出最终合格率，也可以导出 JSONL。checker 启动时会拒绝仍含有一对多 implicit qrel 的旧数据集。

如果希望由多模态模型自动判断每条 `query row -> evidence -> attribute` 是否成立，可运行 10% 稳定抽样审阅：

```bash
conda run --no-capture-output -n MMDD python \
  scripts/mm_joinability_dataset_auto_checker.py \
  --output_dir output_mm_joinability_v15 \
  --provider local \
  --local_base_url http://127.0.0.1:8001/v1 \
  --local_model Qwen3.5-9B \
  --sample_rate 0.10 \
  --seed 13 \
  --cache_path cache/mm_joinability/auto_checker.sqlite3
```

auto checker 对每个 `(row, evidence, attribute)` 做独立的 leave-one-attribute-out 抽取：从原始完整 row 中只移除当前 attribute，模型只看到剩余 row、一个 evidence 和 attribute 名称；模型看不到数据集声称的 value，也看不到 target row。模型只返回 `extracted_value`，checker 再用数据集 builder 相同的归一化规则在本地比较：匹配为 `supported`，非空但不匹配为 `contradicted`，空抽取为 `insufficient`。本地模型优先；只有本地结果为 `contradicted` 或 `insufficient` 时，才会交给 `gpt-5.6-terra` 在关闭思考的情况下做相同的盲抽取，OpenAI 二次筛查最大并发固定不超过 5。最终 verdict 仍由 checker 对二次抽取值做本地比较得到，而不是让模型自己判定。

EntiTables、旧版 WDC 和分阶段 WDC-200K builder 现在也会把这套单属性盲抽取作为模型分析后的强制门禁。批量模型分析结果保存在 `model_attributes`；随后逐个物理移除待检属性，只基于剩余 row 和单条 evidence 重新抽取。只有 `supported` 项会写入下游实际消费的 `attributes`，空抽取、值不一致以及 checker 调用失败都会 fail-closed 过滤，因此不会参与 recovery 覆盖率或 qrels 构造。此次策略同时更新了 prompt/model cache identity，旧的未检查模型输出不会被静默复用。构造阶段先复用 builder 当前配置的文本/图片模型，再用 Luna 做盲抽取初审，并由单独配置的模型进行最终裁决。未完成的 checker 结果不会作为成功缓存复用，恢复运行时会自动重试；只有明确需要纯本地构造时才使用 `--no_auto_check_secondary_openai`。独立 checker 仍可用于抽样复审。

auto-check API profile 推荐写在一个受保护的 JSON 文件中。可从根目录的 `.auto_check_apis.example.json` 开始配置；builder 默认读取 `./.auto_check_apis.json`，也可用 `--auto_check_api_config_file` 指定其他路径，实际配置文件权限必须是 `0600`。profile 名表示供应商及其共享并发预算，不绑定具体模型厂商：

```json
{
  "version": 1,
  "profiles": {
    "gateway_a": {
      "max_concurrency": 12,
      "response": true,
      "initial": {
        "model": "gpt-5.6-luna",
        "base_url": "https://gateway-a.example/v1",
        "api_key": "fake-example-key-a"
      },
      "final_judge": null
    },
    "gateway_b": {
      "max_concurrency": 20,
      "initial": {
        "model": "gpt-5.6-luna",
        "base_url": "https://gateway-b-initial.example/v1",
        "api_key": "fake-example-initial-key-b"
      },
      "final_judge": {
        "model": "grok-4.5",
        "base_url": "https://gateway-b-final.example/v1",
        "api_key": "fake-example-final-key-b"
      }
    },
    "gateway_c": {
      "max_concurrency": 12,
      "initial": null,
      "final_judge": {
        "model": "gemini-3.0-pro",
        "base_url": "https://gateway-c-final.example/v1",
        "api_key": "fake-example-final-key-c"
      }
    }
  }
}
```

带有 `initial` 对象的 profile 参与稳定的初审路由；`initial: null` 表示该供应商只提供最终裁判。`final_judge: null` 的 profile 永远不会收到最终裁判请求。整份配置必须至少有一个初审端点，同一 profile 的两个阶段不能同时为 `null`。若没有任何最终裁判，本地与初审模型的冲突会保持 incomplete 并 fail closed。每个 profile 只有一个自适应并发控制器：从 5 开始，连续成功 20 次后加 1，最高不超过 `max_concurrency`；发生 API 或传输错误时把当前并发减半。如果同一 profile 同时配置初审和终审，即使使用不同 key、URL 和模型，它们仍共享这个 profile 的并发额度。默认使用 OpenAI-compatible Chat Completions；在 profile 上写 `"response": true` 会让两个阶段都走 Responses API，也可以把该字段写进 `initial` 或 `final_judge`，只覆盖对应阶段。两种传输都要求结构化 JSON 输出。builder 运行期间会在每次供应商请求前检查 JSON 文件；只有整份新配置通过读取、权限和结构校验后才会原子切换，新增的初审或终审供应商无需重启即可参与路由。若更新不可读、处于写到一半的状态、权限错误或内容无效，builder 会针对该文件版本输出一次 warning，并继续使用上一份有效的供应商配置。旧的命名 dotenv profile 格式和 `--auto_check_openai_env_file` 仍为兼容保留，但 JSON 是推荐配置。

默认报告写入 `<output_dir>/auto_checker_reviews/`，包括 path 级的本地/二次抽取、query 级覆盖、汇总、失败记录和用于后续数据清理的 `patch_candidates-*.jsonl`。两阶段成功抽取都会按 `(row, evidence, attribute, model identity)` 写入 SQLite cache，失败不会缓存。同一 seed 使用稳定哈希前缀抽样，因此以后把 `--sample_rate` 从 `0.10` 提高到 `0.20` 时会完整复用原 10% 的结果，只审阅新增部分。模型、prompt 或单条抽取输入变化时，对应 cache key 会自动变化。二次筛查从 `OPENAI_API_KEY` 读取密钥并沿用 OpenAI builder 的受限 dotenv 约定；密钥不会写入 cache 或报告。可用 `--no_secondary_openai` 关闭二次筛查。每个模型请求严格只包含一个 `masked row + evidence + attribute name`；同一 evidence 声称能抽取多个 attribute 时会分别请求。

默认情况下，Flask GUI 绑定 `127.0.0.1`，只能在本机访问。给 `stage1_train_models.py`、`stage1_run_all.py`、`run_hitl_training_rounds.py`、`hitl_annotation_app.py`、`stage1_connection_viewer.py` 或 `mm_joinability_dataset_viewer.py` 加上 `--lan` 后会绑定到 `0.0.0.0`，启动日志会同时打印本机 URL 和内网设备可访问的 URL。如果内网 URL 仍然打不开，检查主机防火墙是否放行对应端口，并确认设备在同一个网络里。

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
