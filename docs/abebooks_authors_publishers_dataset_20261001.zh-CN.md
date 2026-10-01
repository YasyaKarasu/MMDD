# AbeBooks 作者与出版社连接数据集

日期：2026-10-01。新副本：`dataset/abebooks_authors_publishers_all5_balanced_20261001`。
父版本：`dataset/abebooks_standalone_all5_balanced_20261001`，保持不变。

内网 Viewer：[打开新版本](http://10.130.141.43:17863/)；
[出版社 implicit 样例](http://10.130.141.43:17863/?q=query_255a9d36c6f4a32c01da)。
服务绑定 `0.0.0.0:17863`，页面和证据图片 HTTP 检查均返回 200。旧 Viewer 保留。

## 结果

加入了 `publisher` 单列连接，保留 `authors` 单列连接。每个查询只指定其中一列；没有改成复合键。

| 划分 | authors implicit | authors explicit | publisher implicit | publisher explicit | 合计 |
|---|---:|---:|---:|---:|---:|
| train | 36 | 34 | 5 | 7 | 82 |
| dev | 12 | 11 | 2 | 3 | 28 |
| test | 12 | 12 | 2 | 2 | 28 |
| 全部 | 60 | 57 | 9 | 12 | 138 |

各划分的 implicit/explicit 总数严格各半；未要求每个连接属性内部也各半。所有查询都是五行，implicit 至少两行有恢复证据。出版社的九个 implicit 查询各有两行证据，其余三行不参与恢复监督。

全部 174 张候选表的列、单元格和行成员均与父版本相同，只重新生成目标 ID。130 张源表、1,289 条源记录和 5,441 个素材全部保留。各查询之间没有复用源行，共使用 690 个不同源行。

本版本不是在旧 132 个查询后简单追加 21 个：构造时先分配出版社视图，再从未使用的源行构造作者视图，重新按源表、重复书目和证据分组划分。最终 117 个作者查询、21 个出版社查询，共 138 个。旧查询与划分保留在父版本及新副本的 `provenance/before_publisher_expansion/` 中。

## 出版社证据与连接判定

出版社匹配规则位于 `src/mmdd_dataset/abebooks_publisher.py`。规范化同一品牌的大小写、标点、常见公司名后缀、地名和版次写法，例如 Morgan Kaufmann 的多个写法以及 Pearson 的多个写法。优先识别具体子品牌，**不把 Elsevier 与 Butterworth-Heinemann、Pearson 与 Prentice Hall 等同**。源字符串和目标单元格仍保留原文；匹配使用规范化后的值。

使用现有目标投影做结构筛查后，选出 89 张相关封面，使用本地 Qwen3.5-9B 盲读，没有提供源出版社答案。随后 Codex 查看全部八张联系表及局部放大图，记录可见文字或出版社标志。89 条提案、图片 SHA256、原始模型输出、逐图复核和拒绝原因均保留在 `work/abebooks_authors_publishers_20261001/` 与数据集 `audit/publisher_evidence_audit.jsonl`。

24 行通过图片与源品牌对应检查，最终九个 implicit 查询使用其中 18 行。包括 MK、AP、BH、Routledge 和 Addison-Wesley 标志的解释在逐图记录中注明。只显示母公司、显示其他品牌、不可读或无法对应书名的图片不写入该行的恢复标签。

本地模型确实出现了错误：将 Elsevier 树形标志读作 MK、将 Digital Press 读作 O'Reilly，或漏读可见标志。最终标签使用图片复核结果，不把这些模型输出当成正确答案。图片复核者是 Codex，且知晓源值，因此这是**模型辅助、知晓源值的图片复核标注，不是独立人工金标，也不是双盲模型一致性标注**。模型原始错误仍在审计中。

结构正例要求按所选属性对整个目标表执行等值连接，五个查询行都获得对应源书目的非空新属性，且任何返回行对都不得扩张到其他书目。不能仅因出版社重叠就标为正例。全部 138 × 174 = 24,012 个 query–target 组合均已判定，产生 164 条正例：作者 142 条、出版社 22 条。

## 验证

出版社的 45 个 implicit oracle 行对均通过结构检查；用实际复核值回放得到 18/18 个正确证据覆盖行对，将这些值在查询行间交换后得到 0/18 个正确行对。这是已筛选数据上的条件回放，只验证值与连接的对应关系，不代表盲测抽取准确率或检索提升。

Stage-1 输入已导出至 `work/abebooks_authors_publishers_20261001/stage1/`，包含各划分的 `edge_lists`、`target_lists` 和 `target_lists.evidence_supervised`。检查了所有查询的五行序列化、属性列索引、正例完整性、仅已核验行进入证据监督、源行不复用、分组不跨划分以及父版本内容哈希一致。

165 项相关测试通过，命令如下（工作目录为隔离目录 `/tmp/abebooks_curation_20261001`）：

```bash
conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_abebooks_standalone.py \
  /home/oycy/MMDD/tests/test_abebooks_rebalance.py \
  /home/oycy/MMDD/tests/test_abebooks_curation.py \
  /home/oycy/MMDD/tests/test_stage1_construction.py \
  /home/oycy/MMDD/tests/test_stage1_pipeline.py \
  /home/oycy/MMDD/tests/test_stage1_retrieval_aligned.py -q
```

完整检查记录：数据集内的 `audit/DELIVERY_VALIDATION.json` 和 `audit/JOIN_COLUMN_VALIDATION.json`。

## 重建

以下命令从隔离工作目录运行。输出目录必须不存在；可换成新的副本路径。需要已有作者盲读产物、历史作者复核缓存和本次出版社盲读／图片复核文件；重建过程本身不调用模型。

```bash
conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/build_abebooks_standalone_dataset.py \
  --dataset /home/oycy/MMDD/dataset/abebooks_standalone_all5_balanced_20261001 \
  --output /home/oycy/MMDD/dataset/abebooks_authors_publishers_all5_balanced_20261001 \
  --proposals /home/oycy/MMDD/work/abebooks_standalone_20261001/local_author_proposals.jsonl \
  --historical-reviews /home/oycy/MMDD/cache/abebooks_mm_joinability/query_recovery_auto_checks.jsonl \
  --publisher-proposals /home/oycy/MMDD/work/abebooks_authors_publishers_20261001/publisher_proposals.jsonl \
  --publisher-reviews /home/oycy/MMDD/work/abebooks_authors_publishers_20261001/publisher_pixel_reviews.jsonl \
  --implicit-rows 5 --explicit-rows 5 --minimum-recovered-rows 2 --balanced

conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/build_stage1_training_data.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_authors_publishers_all5_balanced_20261001 \
  --output-dir /home/oycy/MMDD/work/abebooks_authors_publishers_20261001/stage1

conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/validate_abebooks_standalone_dataset.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_authors_publishers_all5_balanced_20261001 \
  --stage1-dir /home/oycy/MMDD/work/abebooks_authors_publishers_20261001/stage1 \
  --input-snapshot /home/oycy/MMDD/work/abebooks_authors_publishers_20261001/input_snapshot.json
```

## 使用边界

本次未训练检索或连接模型。出版社只有九个 implicit 查询，dev/test 各两个，不能据此宣称出版社任务已获得稳定统计结论。数据仍是历史暴露的小规模语料、共享目标湖与证据库；未核验行的源字段仍作为 oracle 值，不冒充已恢复值。新查询与划分需要新缓存和重新训练；不能直接将旧、新版本分数差解释为方法收益。
