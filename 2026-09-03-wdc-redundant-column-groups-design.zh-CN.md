# WDC 精确冗余列组与多 Target 构造设计

## 1. 文档状态

- 状态：设计方案，尚未实现
- 适用范围：`scripts_old` 中的 WDC/EntiTables multimodal joinability 构造流程
- 目标规模：约 200K source tables
- 设计原则：保留物理列，不做去重；把值完全相同的列视为一个 join family，在 query 层合并，在 target/qrel 层展开

## 2. 背景与问题

当前构造流程按物理列规划 implicit 和 explicit join。对于如下源表：

```text
entity | A       | B       | context
-------|---------|---------|--------
x1     | value-1 | value-1 | ...
x2     | value-2 | value-2 | ...
```

如果 A、B 除属性名外完全相同，现有流程仍可能把它们当成两个相互独立的列：

1. A 和 B 分别占用 join bridge 位置；
2. query context 可能把其中一个 sibling 暴露给模型，造成 leave-one-out 泄漏；
3. explicit 可能为 A、B 分别生成几乎相同的 query；
4. implicit 和 explicit 的候选数、目标数和 qrel 数量被物理列重复放大；
5. 某个物理列没有单独通过模型筛选时，另一列已经足以支持召回，但当前构造仍可能把它们判为不可 join。

这不是简单的重复数据清理问题。A、B 仍然是不同的物理属性，最终数据集中应该保留它们各自的 target 和监督关系；需要合并的是 query-level 的候选规划，而不是删除列或合并 target。

## 3. 目标与非目标

### 3.1 目标

1. 在同一张 source table 内识别精确冗余列组。
2. 一个冗余组只占用一个 query-level bridge slot。
3. 一个 query 可以对应多个物理 target，并为每个 target 生成独立 qrel。
4. materialize 时不强制生成组内所有 target，而是从组内随机选择 1 到组大小个物理列。
5. implicit 和 explicit 的模型 prompt 均保持不变。
6. 粗筛阶段优先复用现有 extraction cache，不因冗余判断增加模型调用。
7. auto_check 阶段确保同组 sibling 不出现在被检查属性的可见输入中。
8. 不改变 canonical 数据集的字段集合和下游读取契约。
9. 保持最终 implicit/explicit query 数量尽可能接近 50%/50%。
10. 把额外计算控制在接近一次已有列值扫描的成本内，不做全局列两两比较。

### 3.2 非目标

1. 不删除、重命名、合并 source table 的物理列。
2. 不把 A/B 替换成一个 canonical 列后只输出一个 target。
3. 不识别近似相似列、语义相似列或跨表相似列。本期只处理精确相等。
4. 不修改 extraction prompt、auto_check prompt 或下游模型输入协议。
5. 不要求下游消费者理解 `group_id`、冗余类型或新的数据结构。

## 4. 冗余定义

### 4.1 精确定义

本期的冗余列组定义为：

> 同一 source table 中，两个或多个不同物理列在相同 source row 顺序上的最终数据集值完全相同；比较时忽略属性名、column name 和 column index。

这里的“完全相同”不是指任意意义上的 normalize 后相同，而是指比较值必须和最终写入数据集的值一致。设最终数据集写入列 `c` 的可见值序列为：

```text
V(c) = [dataset_value(row_0, c), ..., dataset_value(row_n-1, c)]
```

当 `V(A) == V(B)` 且 `A != B` 时，A、B 属于同一个冗余组。

`dataset_value()` 必须直接复用最终 `rows[].cells[].text` 的生成路径：如果最终数据集会对值做清洗、截断或其他规范化，则先执行完全相同的处理；如果最终数据集不做该处理，则必须按原始待输出值逐字节比较。不得为了提高分组数量而额外做大小写折叠、空白折叠、Unicode 归一化、近似匹配或其他只用于冗余判断的 normalize。空值也是序列的一部分，不能因为两列都有空值就跳过对应 row；否则会把行错位的列误判为相同。

### 4.2 分组边界

- 只在一张 source table 内分组，不跨 table 比较。
- 只比较 source row 对齐后的值，不按排序后的值集合比较。
- 不要求列名不同；分组单位是物理列 index。
- 全空列即使 hash 相同，也仍然要经过现有 `min_non_empty_ratio`、`min_rows` 等资格检查；冗余检测本身不能让无效列成为 target。
- hash 只用于高效分桶。实现应以完整的 row-aware digest 作为等价键；若需要极端保守，可在同 digest 桶内用已缓存的短签名再次确认，但不得退化成全列两两比较。

### 4.3 Entity 列的处理

分组检测可以覆盖 entity 列，以便识别例如 `name`、`headline`、`mainentityofpage` 这类完全相同的 entity alias。但规划阶段需要额外约束：

- 含当前 entity 列的 group 不得作为 implicit hidden bridge。因为 entity 必须可见，隐藏 group 的其他成员会被 entity alias 泄漏。
- 含 entity 列的 group 默认不参与 multimodal recovery 扩展。
- 如果现有 explicit fallback 选择了其中的非 entity 成员，仍按显式 visible join 的原有规则处理；target 扩展只允许使用非 entity 成员，不把 entity 物理列重复投影成 target。

这个限制只防止把 entity alias 当成多模态恢复能力，不影响普通 source table 中的属性列冗余组。

## 5. 目标语义

### 5.1 核心模型

冗余组 `G = {A, B, ...}` 是一个 query-level join family：

```text
一个 query
    -> target_A
    -> target_B
    -> ...
```

每个 target 仍然保留自己的物理列 index、列名、target table ID、chain ID 和 qrel。只有 query 的可见内容和候选规划按 group 合并。

### 5.2 Implicit

对于属性组 `G = {A, B}`：

```text
Query   = entity + query_context
          # A、B 全部隐藏

TargetA = A + target_context_A
TargetB = B + target_context_B

qrels:
Query -> TargetA, reason=model_recoverable_join_column
Query -> TargetB, reason=model_recoverable_join_column
```

约束：

- query context 不得包含 A 或 B 中任何一个成员；
- `target_context_A` 和 `target_context_B` 可以不同，并且应优先不同；
- 模型恢复一次 group-level value 即可支撑多个物理 target；
- 每个目标成员对应一条 qrel；
- query table 只创建一次，不能因为 A/B 分别 materialize 而创建两个可见内容相同的 query；
- 如果不同 hidden group 产生完全相同的 visible query，继续复用现有 visible-query fingerprint 合并逻辑，并把 target/qrel 追加到同一个 query。

### 5.3 Explicit

如果组内 A 被选为 visible join：

```text
Query   = entity + A + query_context
TargetA = A + target_context_A
TargetB = B + target_context_B

qrels:
Query -> TargetA, reason=explicit_visible_join_column
Query -> TargetB, reason=explicit_visible_join_column
```

约束：

- 一个 group 只生成一个 explicit query-level candidate；
- A 是可见列时，A 必须包含在本次 target 子集中，避免产生“query 已经显示 A、但没有对应 target_A”的不自然样本；
- target 子集仍可以只包含 1 个成员，因此最小情况是 `Query -> TargetA`；
- `target_context_A` 和 `target_context_B` 可以不同，并且应优先不同；
- 当随机 target 数大于 1 时，再从其余 sibling 中抽样；
- B 不进入 query context，否则 TargetB 会被直接泄漏；
- A/B 不分别生成两个 query。

## 6. 整体流程

```text
读取 source table
    |
    v
复用已有列值扫描，计算每列 row-aware digest
    |
    v
按 digest 分桶，得到冗余组（不写入 canonical output）
    |
    +--> 粗筛：原 candidate names、原 prompt、原 extraction cache key
    |
    v
把通过粗筛的物理列映射到 group-level bridge candidate
    |
    +--> implicit：group 作为一个 hidden bridge，全部成员从 query context 排除
    |       |
    |       +--> auto_check 输入预处理时屏蔽整个 group
    |       +--> 接受后随机抽样 1..|group| 个 target members
    |
    +--> explicit：group 作为一个 visible bridge，确定一个 visible member
            |
            +--> auto-check 不改变 prompt；不把 sibling 放入 query context
            +--> visible member 强制进入 target 子集
            +--> 再随机抽样 sibling target members
    |
    v
按现有字段写 query_tables / data_lake_tables / qrels / recoveries
    |
    v
按 query 数而非 target/qrel 数平衡 implicit 与 explicit
```

## 7. 冗余组检测算法

### 7.1 推荐实现

在现有 `table_column_values(source_table)` 产生 `values_by_column` 时同步计算 digest，不再新增一次完整的 source row 扫描。digest 的输入必须是最终数据集列值生成路径的结果，而不是另写一套 normalize 逻辑。每个物理列只保留固定大小的状态：

```text
column_index
row_count
sha256(row_0_value_with_length, ..., row_n_value_with_length)
```

长度前缀或明确的不可混淆分隔符是必需的，避免 `['ab', 'c']` 与 `['a', 'bc']` 得到同一串字节。推荐把 row index/row count 绑定到顺序协议中，而不是把所有值连接成无边界字符串。

伪代码：

```python
def exact_redundancy_groups(values_by_column):
    buckets = {}
    for column_index, values in values_by_column.items():
        digest = sha256()
        digest.update(encode_u64(len(values)))
        for row_index, value in enumerate(values):
            dataset_value = serialize_cell_for_dataset(value)
            digest.update(encode_u64(row_index))
            digest.update(encode_u64(len(dataset_value.encode("utf-8"))))
            digest.update(dataset_value.encode("utf-8"))
        buckets.setdefault(digest.hexdigest(), []).append(column_index)

    return [
        sorted(member_indices)
        for member_indices in buckets.values()
        if len(member_indices) >= 2
    ]
```

实际代码应复用项目现有的稳定 hash/清洗工具，并使用稳定排序保证同一 seed 下结果可复现。

`serialize_cell_for_dataset()` 只是上述规则中的示意名称，实际实现必须调用当前 target/query 投影最终写入 `rows[].cells[].text` 的同一个序列化函数；不能额外引入一个只服务于冗余判断的 normalize 函数。

### 7.2 复杂度

对一张有 `R` 行、`C` 列的表：

- 时间复杂度：`O(R * C)`；
- 额外内存：`O(C)` 个 digest 和少量组元数据；
- 额外磁盘：默认 `0`，不保存全量列值或 pairwise 比较矩阵；
- 全部 200K tables：`O(sum(R_t * C_t))`，且这部分工作可以直接并入已有列值扫描。

这比 `O(C^2 * R)` 的列两两比较更适合 200K 规模。由于 WDC source table 通常已经需要生成 `values_by_column`、column profile 和投影表，digest 计算的主要成本是对已存在字符串再做一次顺序 hash，不能引入网络请求或模型请求。

### 7.3 什么时候计算

分两阶段兼容现有代码：

1. **短期实现位置**：`build_table_join_records()` 内复用已经构造的 `values_by_column`，计算一次 group map。
2. **规模优化位置**：如果 WDC structural stage 已经一次性读取 source rows，则在 `_read_table_once()` 的同一行循环中更新 digest，把结果放入构造内部的 structural cache 或随候选对象传递；canonical `source_tables` 不要求新增字段。

不建议在 materialize 阶段重新打开源文件做列比较，也不建议为每个候选 join column 单独计算 group。

## 8. Group-level candidate 规划

### 8.1 内部对象

实现内部可以使用如下对象，但不写入 canonical dataset：

```text
RedundantColumnGroup
    group_key              # table_id + digest，稳定且只在 table 内有效
    member_column_indices   # 物理列 index，升序
    member_column_names     # 仅供规划和日志
    canonical_member        # 默认最小 index
    contains_entity         # 是否含当前 entity column
```

`qualified_cols` 仍可保留物理列级结果，group planner 只负责将它们映射到一个 bridge candidate。这样可以继续使用现有的 recovery ratio、query row selection 和审计逻辑。

### 8.2 Implicit planner

1. 先按现有流程得到物理列级 `qualified_cols`。
2. 将 qualified physical columns 按 redundancy group 聚合。
3. 一个 group 只保留一个 group-level variant；选择 representative 时优先使用已经通过 recovery 的成员，若有多个则按 `recovered_value_ratio`、再按 column index 稳定排序。
4. `context_columns()` 的 excluded 集合使用 `{entity_col} ∪ 所有 emitted group members`，而不是只排除 representative。
5. `multi_attribute_context_layout()` 的 bridge slot 数按 group 计算。
6. query context 仍按现有随机策略生成；target context 先得到一个不含 entity/group members 的候选池，再按 target member 使用独立稳定 seed 抽样，允许 `target_context_A`、`target_context_B` 不同。
7. group 通过 auto_check 后，随机选择 target members 并为每个 member 建立 target/qrel。

### 8.3 Explicit planner

1. `_explicit_join_candidate_columns()` 先按现有非空率、行数规则生成 physical candidates。
2. 把属于同一 group 的 candidates 合并为一个 group-level candidate。
3. group 内 visible member 选择优先级：
   - 现有 explicit candidate 资格最高者；
   - 若资格相同，使用 seed + source table ID + group key 的稳定 hash；
   - 最后以 column index 作为 tie-breaker。
4. `_explicit_join_context_partition()` 的 excluded 集合按 group 全部成员计算。
5. 每个 group 最多生成一个 explicit query；现有 `max_query_tables_per_source_table` 限制 group-level query 数。
6. target context 按 target member 独立抽样，并优先让不同 member 使用不同 context 集合；不能把 A/B 或其他 group member 放入任一 target context。
7. target 抽样时把 visible member 放入 mandatory set，再从 sibling 中补足随机数量。

### 8.4 Target context 的差异化策略

target context 不应简单地对所有冗余 member 复用同一个列列表。推荐流程如下：

1. 先从 entity 列、整个 redundancy group 和其他已禁止列中排除，得到 ordinary target-context pool。
2. query context 仍只生成一次；target context 从与 query context 不冲突的候选池中选择。
3. 对每个 target member 使用 `global_seed + source_table_id + group_key + member_column_index` 生成独立稳定排序或随机数。
4. 如果候选池足够大，优先选择与已经生成的其他 target context 集合不同的列子集，并以集合差异最大、稳定 hash 最小作为 tie-breaker。
5. 如果现有 context 上限要求使用全部候选列，或者候选池只有 0/1 列，则允许复用；此时差异不可构造，不应为了制造差异加入原本不允许的列。

因此，A/B 的 join value 一样，但 TargetA 和 TargetB 可以拥有不同的辅助属性。只有在源表没有足够的非 group 属性时，才接受 target context 复用。

## 9. Materialize 的随机 target fanout

### 9.1 规则

对大小为 `k` 的冗余组，目标 target 数 `m` 满足：

```text
m = randint(1, k)
```

抽样必须是：

- 无放回；
- 同一 source table、同一 group、同一 seed 下可复现；
- 不依赖线程完成顺序；
- 不增加模型调用。

推荐使用：

```text
seed = stable_hash(global_seed, split, source_table_id, group_key)
```

implicit 直接从全部 group members 抽 `m` 个；explicit 先把 visible member 放入集合，再抽 `m - 1` 个 sibling。若 `k == 1`，自然生成一个 target。target context 的抽样独立于 member 抽样，但使用同一套稳定输入和 member-specific seed。

### 9.2 为什么在 group-level 随机抽样

抽样应发生在 query-level group 被接受之后，而不是在每个 physical column candidate 上分别抽样。这样可以保证：

- 同一个 query 不会因 A/B 产生重复 materialization；
- 同一 row view 的多个 target 使用一致的 group 成员集合；
- 同一 group 的不同 target 可以使用不同的 target context；
- target fanout 不会改变 implicit/explicit query 配额；
- 多 target 的 qrel 和 recovery fanout 可以一次完成。

### 9.3 Recovery 的 fanout

如果粗筛/auto_check 只验证了 A，但 A/B 的值完全一致，不需要为 B 再发起模型调用。采用如下规则：

1. 以通过 auto_check 的 group-level recovery evidence 作为证据来源。
2. 对每个被抽中的 target member，复制一条逻辑 recovery path。
3. 复制时只改写现有 target-specific 字段：target ID、target row IDs、path ID、recovered attribute 的 physical column index/name；证据 asset 和模型结果复用原记录。
4. 对同一 query row + asset + final dataset value 先去重，再向多个 target fanout，避免同一个 asset 因 A/B 的粗筛结果重复写入。

这样既表达了 `Query -> TargetA` 和 `Query -> TargetB`，又不把模型调用数和证据内容按 group size 成比例放大。

## 10. Prompt 与缓存策略

### 10.1 粗筛阶段

粗筛不修改 prompt：

- `candidate_attribute_names` 继续使用物理列名列表；
- 继续使用现有 leave-one-attribute-out extraction prompt；
- 不把 redundancy group 放进 prompt；
- 不把 group ID 加入 extraction cache key；
- 已有 model extraction cache 可以直接复用。

原因是粗筛本来就是高召回、低精度阶段，冗余组只在其输出之后参与 group-level 规划。若因为加入 group 信息而改变 cache key，会导致已有 200K 规模缓存大面积失效，收益不值得。

### 10.2 Auto-check 阶段

auto_check prompt 也不修改，只修改传给现有 checker 的输入对象：

- 当前候选属性为 A 时，从 `query_row_attributes` 同时移除 A 和其 group sibling；
- 当前候选属性为 B 时，同样移除整个 group；
- entity 列仍按现有规则保留，除非它属于 entity-containing group 的不参与 recovery 分支；
- prompt 继续把剩余属性当作普通可见属性。

也就是说，prompt 文本不变，mask 由调用前的 row attribute projection 完成。

### 10.3 Auto-check cache key

必须避免复用“只屏蔽 A、但仍暴露 B”的旧 auto-check 结果。推荐让 auto-check 的有效 cache identity 包含：

```text
asset_id
asset_type
masked_row_attributes       # 已移除整个 group
attribute_name
claimed_value
auto_check_schema_version
```

实现上可以复用现有 `query_recovery_auto_check_key()` 的结构，但传入 group-masked row attributes，并 bump auto-check cache schema version。粗筛 extraction cache 不 bump；prompt version 也不需要 bump，因为 prompt 文本没有变化。

旧 auto-check cache 的处理原则：

- 如果旧 key 无法证明使用了完整 group mask，不得直接作为最终 verdict；
- 可以保留旧文件以便其他任务使用，但本次构造使用新的 schema/key namespace；
- 新运行中相同 asset、相同 masked row 和相同 claimed value 的检查仍可跨 query 复用。

## 11. 输出字段兼容性

本设计不新增下游必须读取的字段，不要求修改下游数据集消费者。

### 11.1 保持不变的 canonical artifacts

以下 artifact 的字段集合保持现有契约：

- `source_tables`
- `query_tables`
- `data_lake_tables`
- `qrels.jsonl`
- `evidence_recoveries`
- `bridge_assets`
- `entities`
- `table_asset_links`

尤其不新增 `group_id`、`redundancy_members`、`fanout_count` 等字段到 query/target/qrel 记录中。

### 11.2 复用已有字段表达多 target

现有字段已经足够表达新语义：

- query 的 `target_table_ids`：放入该 query 的全部 target IDs；
- query 的 `hidden_attributes`：按 qrel 保留实际物理 hidden member 的描述；
- target 的 `join_col` / `join_col_name`：分别指向其物理 member；
- qrel 的 `target_table_id`、`data_lake_table_id`：每个 target 一条关系；
- qrel 的 `join_attribute`：记录当前 qrel 对应的物理列；
- `chain_id`：可按 physical target member 生成稳定 ID，query 的 `chain_ids` 继续容纳多个 chain；
- recovery 的 `target_row_ids`、`path_nodes`、`path_id`：按 target fanout 生成现有格式的 target-specific 记录。

因此，下游只会看到“一个 query 有多个正 target”，不需要知道这些 target 是否来自冗余列组。

### 11.3 构造审计信息

如果需要审计冗余检测，可将 group 数量、fanout 分布、auto-check mask 统计写入现有 `stats.json` 或 `dataset_manifest.json` 的构造说明段；这些信息不是 canonical 数据集输入，下游无需读取。若严格要求字段完全不变，也可以只写入内部 cache/log，不改变最终输出文件。

## 12. Implicit/Explicit 50%/50% 平衡

### 12.1 配额单位

平衡单位必须是 **query table 数**，不是 target table 数、qrel 数或 recovery 数。

例如，一个 implicit query fanout 到 3 个 target，仍然只计 1 个 implicit query；一个 explicit group query fanout 到 2 个 target，也仍然只计 1 个 explicit query。

### 12.2 推荐流程

1. 先完成 implicit 的 group-level planning 和 materialization，得到每个 split 的 implicit query count。
2. explicit candidate enumeration 按 group 去重，一个 group candidate 对应一个 query slot。
3. 在每个 split 内，以 implicit query count 为目标选择 explicit group candidates。
4. 再 materialize 选中的 explicit groups；target fanout 不参与配额。
5. 最终验证每个 split 的：

```text
explicit_query_count == implicit_query_count
```

如果某个 split 的显式 group candidate 不足：

- 继续沿用现有 source replacement / candidate selection 机制寻找新的 source table；
- 不通过重复同一 group、重复同一 query 或增加 target fanout 来填补 query slot；
- 若仍无法填满，记录 `unfilled_slots`，保留现有 fail-closed 行为。

### 12.3 对现有 `match_implicit` 的影响

现有 `match_implicit` 逻辑可以继续复用，但 candidate identity 要从 physical `join_col` 调整为 group-level identity。选中后，`rebuild_selected_explicit_join_candidates()` 按选中的 groups 重新分配 context；每个 group 只 materialize 一个 query，目标列再按随机 fanout 展开。

## 13. 并行与性能设计

### 13.1 不建议并行的部分

同一张表内部的列 digest 计算不建议拆成每列线程：

- 任务太小，线程调度开销可能高于 hash 成本；
- 会产生更多临时对象和内存带宽竞争；
- 不利于保持现有 source table 的顺序处理。

正确的优化是一次 row loop 更新多个列 digest，而不是对列做两两比较或每列单独扫描。

### 13.2 可以并行的部分

如果 structural stage 需要单独计算并持久化 group metadata，优先按 source file/table 做并行：

- 使用 bounded `ProcessPoolExecutor` 处理 gzip 解压、JSON 解析和 row-aware digest；
- worker 数默认取 `min(max(1, cpu_count - 1), configured_max_workers)`；
- 每个 worker 只返回紧凑的 source table ID、digest 和 member indices，不返回全量 rows；
- 主进程按稳定的 source table 顺序合并结果并写 shard，避免输出乱序导致不可复现；
- worker 不直接写共享 JSONL、SQLite 或 model cache，避免锁竞争和 cache 损坏。

如果 group detection 直接放在已经读取完 source table 的 `build_table_join_records()` 内，则无需额外进程池；调用一次 `values_by_column` 后在主线程完成分桶即可。此时整体瓶颈仍然是模型请求，冗余判断只增加轻量 CPU 工作。

### 13.3 与模型并发的边界

不要把整个 `build_table_join_records()` 并行化到多个 worker：

- extraction cache、auto-check cache、sharded writers 和进度状态已有共享并发模型；
- 全流程并行会让 cache key 去重、writer 顺序和资源回收复杂化；
- group detection 本身不值得承担这些风险。

推荐的并发层次为：

```text
source file/table level: structural parsing and digest，可 bounded process parallel
asset fetching: 复用现有 Wikipedia/web worker
model extraction: 复用现有 text/image endpoint concurrency
auto_check: 复用现有 local/remote reviewer pool
group planning/materialize: 主进程轻量 deterministic pass
```

### 13.4 磁盘占用

默认不新增全量列值 sidecar。最低磁盘方案是：

- digest 只存在于当前 table 的内存对象中；
- structural stage 若需要断点恢复，只持久化每 table 的 group descriptor，而不是全列 values；
- descriptor 只包含 source table ID、column indices、digest、版本和 seed 相关参数；
- auto-check cache 复用已有 cache 目录，不复制 evidence 内容；
- canonical output 不保存 group metadata。

如果 200K 规模需要跨阶段复用 group detection，推荐使用一个紧凑 SQLite/JSONL sidecar，并按 source table ID 建索引；其空间复杂度为 `O(number_of_redundant_groups + number_of_members)`，而不是 `O(number_of_rows * number_of_columns)`。

## 14. 稳定性、版本和断点续跑

### 14.1 版本标识

建议分别维护：

- structural group descriptor version：用于 group digest 协议变化时失效；
- auto-check cache schema version：用于完整 group mask 变化时失效；
- 不修改 `PROMPT_VERSION`，因为 prompt 文本未变；
- canonical dataset format 不变，不需要下游迁移版本。

如果 group metadata 进入 structural stage 的缓存 fingerprint，必须把 digest 协议、清洗规则版本和 entity-group policy 纳入 fingerprint。

### 14.2 确定性

下列结果必须只由稳定输入决定：

- column digest；
- group member 排序；
- visible member 选择；
- target fanout 数；
- target member 抽样；
- target/query/chain/recovery IDs。

不得使用线程完成顺序、Python 默认 hash 随机化或未排序的 set iteration 参与 ID 和抽样。

## 15. 测试计划

### 15.1 Group detector

1. A/B 值逐行完全一致、列名不同时分到同一组。
2. 只有一行不同不分组。
3. 行顺序不同不分组。
4. 空值位置不同不分组。
5. 全空列不会绕过已有资格过滤。
6. 同一 table 的 group 不会跨 table 合并。
7. digest 计算结果与直接比较最终数据集值序列的结果一致。

### 15.2 Implicit

1. A/B exact group 只生成一个 query。
2. query 不包含 A 或 B。
3. target 只从 1 到 2 个 member 中随机选择。
4. 一个 query 对每个被选 target 生成一条 qrel。
5. 不同 target 优先使用不同的 target context；没有足够候选列时才允许复用。
6. auto-check masked row 同时移除 A、B。
7. A 通过 recovery 而 B 未通过 physical extraction 时，仍可以生成 TargetB，但不增加模型调用。
8. 多个 row view 不会重复生成同一个 query。

### 15.3 Explicit

1. A/B group 只生成一个 query-level candidate。
2. query 显示 A 时，target 子集一定包含 A。
3. B 不会进入 query context。
4. 不同 target 优先使用不同的 target context；没有足够候选列时才允许复用。
5. target fanout 为 1 时仍能生成合法显式样本。
6. 同一 source table 的多个 group 不会互相暴露 sibling。

### 15.4 Schema 和配额

1. canonical record 的字段集合与旧实现一致。
2. `target_table_ids`、`hidden_attributes` 和 qrels 能表达多 target。
3. fanout 不改变 implicit/explicit query 计数。
4. 每个 split 的 implicit/explicit query 数相等或只因 candidate shortfall 留下明确的 unfilled slots。
5. 旧 extraction cache 可以命中；旧 auto-check 结果在无法证明完整 mask 时不会被错误复用。

### 15.5 实际样例验证

使用 `st_wdc_72bb6f163a12a3d4` 做离线回归，重点确认：

- 完全相同的属性列不会产生多个重复 query；
- 一个 query 可以关联多个对应物理 target；
- `name`/`headline`/`mainentityofpage` 这类 entity alias 不会成为 implicit hidden bridge；
- `author`/`publisher` 这类完全相同属性列可以按 group fanout；
- `datepublished`/`datemodified` 若存在差异，不会被本期 exact policy 错误合并。

## 16. 实施顺序

1. 增加纯函数 `exact_redundancy_groups(values_by_column, entity_col=None)`，先完成单元测试。
2. 将 group detection 接入现有 `build_table_join_records()` 的 `values_by_column` 生命周期。
3. 修改 implicit planner：group-level slot、全 sibling 排除、target fanout 和 recovery fanout。
4. 修改 explicit planner：group-level candidate、visible member mandatory target、member-specific target context、group-level balance。
5. 修改 auto-check 调用前的 row attribute projection 和 cache key/schema；不改 prompt 文本。
6. 增加 canonical schema key-set 回归测试和 50/50 split-level invariant。
7. 用实际 WDC 表做离线 materialize 验证，再进行小规模 2K/20K smoke run。
8. 统计 group fanout、额外 CPU 时间、extraction cache hit rate、auto-check cache hit rate 和最终 implicit/explicit 比例，再决定是否把 digest 下沉到 structural stage。

## 17. 验收标准

本设计实现完成后，至少应满足：

```text
同表内 exact duplicate columns -> one query-level bridge
one accepted group -> 1..|group| target members
one query -> multiple target IDs and qrels
implicit query hides every group member
explicit query exposes at most the selected visible member
auto_check sees no group sibling leakage
coarse extraction prompt and cache contract unchanged
canonical output field set unchanged
implicit/explicit query counts remain approximately 50%/50%
redundancy detection has no O(C^2) pairwise scan
additional persistent storage is zero by default
```
