SELECT 7 AS version_order, 'v7' AS version, 1318 AS queryable_sources,
       1318 AS query_tables, 1318 AS qrels, 0.5 AS min_ratio,
       '全局 replacement；每个 source 最多一个 query' AS major_change
UNION ALL
SELECT 8, 'v8', 1680, 2170, 2232, 0.5,
       '宽表多属性 query variants 与 visible-query merge'
UNION ALL
SELECT 9, 'v9', 2648, 3341, 3445, 0.4,
       '恢复门槛从 3/5 降为 2/5';
