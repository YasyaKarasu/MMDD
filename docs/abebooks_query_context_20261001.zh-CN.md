# AbeBooks 查询补充可见上下文

新副本：`dataset/abebooks_authors_publishers_context2_all5_20261001`。
父版本：`dataset/abebooks_authors_publishers_all5_balanced_20261001`，内容未改动。

138 个查询现在全部为 **5 行、2 列**。原来的 69 个单列 implicit 查询补充真实源字段：67 个增加 `publication_year`，另 2 个作者连接查询增加 `publisher`。其中一个年份字段有一行原始缺失值，保留缺失，不填造值。69 个 explicit 查询已经是两列，输入及 ID 保持不变。

| 查询类型 | 可见列 | 数量 |
|---|---|---:|
| implicit，作者连接 | title + publication_year | 58 |
| implicit，作者连接 | title + publisher | 2 |
| implicit，出版社连接 | title + publication_year | 9 |
| explicit，作者连接 | title + authors | 57 |
| explicit，出版社连接 | title + publisher | 12 |

总数仍为 138，train/dev/test 为 82/28/28，每个划分显隐各半。查询行成员、隐藏连接列、恢复覆盖行、164 条 query–target 正例和全部 174 张目标表保持不变。仍有 117 个作者连接、21 个出版社连接；没有增加素材、改变标注或重新划分样本。

## 字段选择与验证

优先选择出版年份，要求至少三行具有有效值；若不适合，再尝试出版社，且不得把隐藏出版社重新显示出来。逐查询检查所有可见单元格中的隐藏值提示，并重新判定原连接能否补充非空目标属性。

另外，针对全部候选表检查新增字段单独执行等值连接的结果：如果至少覆盖两个有用查询行且不产生异书扩张，则拒绝该字段。两处年份因此被拒绝，改用出版社。重复书名的跨记录匹配按潜在对应处理，避免利用身份未判定来宣称没有直接连接。该检查仅针对所述单列等值连接，不能证明检索难度完全不变或消除所有语义线索。

补列后重新执行全部 24,012 个 query–target 判定，正负标签没有变化。Stage-1 全量检查通过，确认两列和五行均进入实际模型序列化输入，只有原已核验的行参与恢复监督。原数据集和素材的内容哈希保持一致。

代码：`src/mmdd_dataset/abebooks_query_context.py`、`src/enrich_abebooks_query_context.py`。共享查询检查同时扩展为检查所有可见字段，而非只检查书名。

验证产物位于新副本的 `audit/DELIVERY_VALIDATION.json` 和 `audit/query_context_changes.jsonl`。168 项相关测试通过：

```bash
conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_abebooks_standalone.py \
  /home/oycy/MMDD/tests/test_abebooks_rebalance.py \
  /home/oycy/MMDD/tests/test_abebooks_curation.py \
  /home/oycy/MMDD/tests/test_stage1_construction.py \
  /home/oycy/MMDD/tests/test_stage1_pipeline.py \
  /home/oycy/MMDD/tests/test_stage1_retrieval_aligned.py -q
```

## 重建及查看

从隔离目录 `/tmp/abebooks_curation_20261001` 执行以下命令，输出目录必须不存在：

```bash
conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/enrich_abebooks_query_context.py \
  --dataset /home/oycy/MMDD/dataset/abebooks_authors_publishers_all5_balanced_20261001 \
  --output /home/oycy/MMDD/dataset/abebooks_authors_publishers_context2_all5_20261001

conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/build_stage1_training_data.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_authors_publishers_context2_all5_20261001 \
  --output-dir /home/oycy/MMDD/work/abebooks_query_context_20261001/stage1

conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/validate_abebooks_standalone_dataset.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_authors_publishers_context2_all5_20261001 \
  --stage1-dir /home/oycy/MMDD/work/abebooks_query_context_20261001/stage1
```

Viewer 已切换到新副本，内网入口仍为 [http://10.130.141.43:17863/](http://10.130.141.43:17863/)。补列的 implicit 查询重新生成了 ID；旧、新 ID 映射位于 `audit/query_context_changes.jsonl`。请从首页选择查询。训练输入也已重新导出；本次未运行模型训练，历史暴露及模型辅助标注等限制继续适用。
