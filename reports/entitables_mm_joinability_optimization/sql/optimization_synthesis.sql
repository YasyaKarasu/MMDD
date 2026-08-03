SELECT 1 AS priority_order, 'P0' AS priority,
       '对齐预采样资格：active entity rows ≥ 5' AS lever,
       '13,008/40,000 槽位结构性必失败；实测正例为 0' AS evidence,
       '释放 32.5% 最终槽位和对应证据/模型预算；净增需 replay 实测' AS expected_impact,
       '若只按高行数筛选会改变数据分布，需分层采样' AS risk
UNION ALL SELECT 2, 'P0',
       '修短 token matcher + 保存 failed-column near-miss',
       '49 条短非精确 recovery；15 个 qrel 的 ≥2 安全行不足',
       '先提高 label precision，并让后续阈值/alias 反事实可离线计算',
       '合理缩写可能被误删，必须按属性类型和 alias 审核'
UNION ALL SELECT 3, 'P1',
       'Train-only 第二个 disjoint 5-row view，cap=2',
       '1,321/3,445 qualified attributes 可支持至少第二个不相交视图',
       '链路变体 3,445→4,766（+38.3%）；unique query 需生成后去重',
       '同源相关性和属性头部偏置上升；dev/test 不应扩增'
UNION ALL SELECT 4, 'P2',
       'drop_probability 0.5→1.0，保持 2 rounds',
       '当前消费 78,251 候选；容量上限 120,000；总体命中率 3.38%',
       '粗略外推约 +1.4k queryable source；候选分析量约 +53%',
       '模型/抓取成本明显增加，且条件命中率可能下降'
UNION ALL SELECT 5, 'P2',
       '证据感知分层采样与属性配额',
       '实体行越多命中率越高；Top-5 属性占 49.5% qrels',
       '提高每次模型调用的 query yield，同时控制 Year/Venue 等头部属性',
       '不加保留概率/权重会破坏总体代表性'
UNION ALL SELECT 6, 'P3',
       '阈值 0.4→0.2，允许单行恢复',
       '984/2,648 queryable source 已处在恰好 2 行的边界',
       '可能大幅增量，但当前产物无法精确反事实估计',
       '标签强度显著下降，不建议在 matcher QA 前采用';
