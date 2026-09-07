# R10 实验方案：连接性分数学习、抗弱路径累积的聚合与 evidence 桥接验证

日期：2026-09-07。状态：后续实验执行方案，替代 [r9](stage1_optimization_r9_plan_20260901.md)。本文定义待实现、待运行的实验，不代表训练已经启动或取得结果。

产物根目录：`work/stage1_optimization_r10_20260907/`。新增实现放在 `src/mmdd_stage1/`、`src/mmdd_stage2/` 与 `src/` 入口中，配置放在 `configs/`，实验目录只放运行产物。沿用 `MMDD` conda 环境。

## 1. 本轮研究问题与范围

主线保持 [方案.md](方案.md) 的原始承诺：缺失属性的 query examples，通过可组合的多模态连接性检索 evidence，再定位、补值，最终发现真实可 join 的目标表。底层 Qwen3-VL-Embedding 冻结；Student 每种类型的 P 可训练，有序关系 R 可训练；在线 Stage-1 仍只运行可索引 Student，允许 direct 与单 evidence 中介的 0/1 跳。

本轮同时研究两个互补问题：

1. **训练侧区分度。** Raw embedding 的相似度不是连接性，0.5 左右的无关边和 0.7-0.8 的相关边可能没有足够差距。通过局部监督、P+R、蒸馏和难负例，让有证据支持的边获得高连接置信度、确认无效的边接近 0。
2. **聚合侧数量偏置。** 即使边分数改善，也不能让大量弱、重复或不支持当前连接的路径仅凭数量超过少量可靠路径；同时应允许不同 query rows 上的有效证据提供互补收益。

因此不能只扫 aggregation，也不能只增加 edge loss。必须保留“只改训练 / 只改聚合 / 两者一起改”的交叉实验。按用户补充，**log-mean-exp 是本轮优先实现和评测的正式聚合候选**，先用它验证去掉纯数量奖励的效果，再比较更强的路径筛选与行覆盖设计。

### 保留与取消的约束

- 保留 Teacher-Student、有向双线性检索、单对象 ANN、0/1 跳、例子行、text+image 和二阶段属性恢复。
- r9 不再单独执行。已完成的代码修复可以继承，未完成的 P、edge、mining 消融纳入本轮重新设计。
- `weighted_rrf e=0.05` 仅作为历史工程对照，不再决定主实验 checkpoint 和是否开展后续实验。
- 不用在线 direct Teacher 重排的收益替代 Student 或 evidence 路径的收益；在线重排列为可选扩展对照。
- 不以一个点数门槛宣称某个研究方向“已证伪”，也不把单次训练失败解释为模型函数类不足。
- 主轮不增加 encoder 微调、额外跳数或非线性 P；先把原始线性 P+R 与证据聚合测清楚。

## 2. 分数定义：接近 0/1 的对象是什么

对关系类型 r，区分三个量：

```text
c(a,b) = cosine(z_a, z_b)                    # raw 相似度，只作基线
s(a,b) = (P_type(a) z_a)^T R_r (P_type(b) z_b) # Student 原始 logit，无界
ell(a,b) = a_r * s(a,b) + b_r, a_r > 0
p(a,b) = sigmoid(ell(a,b))                    # 可训练的连接置信度
```

目标是确认无效边的 p 接近 0、有效边的 p 接近 1，**不强求双线性 s 本身落在 [0,1]**。p 未通过独立校准检查前称“置信度”，不称真实概率。模糊边允许居中；不通过过度放大 logit 制造看似漂亮的 0/1 直方图。

`a_r=softplus(alpha_r)+epsilon`，按有序类型对学习单调仿射变换。对同一 source 和 destination type，单调变换保持 s 的排序，因此仍按 `(R_r^T u_a)^T u_b` 做 ANN，取回候选后再计算 p，不破坏单向量索引。

必须区分“只学 a/b 的分数重标定”和“P/R 改变排序”的贡献。只调 sigmoid 温度不会改善同关系内的 AUROC 或 ANN 排名；跨模态合并排名可能改变，需单列报告。

### 局部标签的语义

| 对象或路径 | 正例含义 | 不能直接当作确认负例的情况 |
| --- | --- | --- |
| Q -> E | E 对 Q 中至少一个实体有可验证的属性恢复支持 | 不支持当前 T，但能支持 Q 的另一个合法桥接属性 |
| E -> T | E 提供的属性和值能与 T 的某个合法桥接列连接 | 只是不属于该 evidence 原先的 source table |
| Q -> E -> T | 两跳能组成同一个实体-属性-值的有效连接 | 只有局部语义相近，缺少同一属性的支持 |
| Q -> T | 对目标检索是潜在可连接目标；显式可直接 join 单独标注 | implicit 正目标不能被误称为已具备可见 join 条件 |

现有 recovery、qrel、行及列映射作为离线监督来源，provenance 不进入模型可见输入。局部正边不保证整条路径一致，因此另保留路径层监督。未标注不等于无效；Teacher 分数本身也不能充当确认负标签。

## 3. Task A：数据、输入与评价协议

### A1. 固定实际数据快照

截至制定本计划时，EntiTables 最新候选目录为 `output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9/`，已存在 manifest；WDC 候选目录为 `output_wdc_webtable_200000_qwen35_local_autocheck_v9/`，尚未发现根目录 manifest。此处仅记录观察，运行时仍须确认完成状态和产物完整性。

数据集 v9 与已取消的实验 r9 没有执行依赖。EntiTables 完成审计即可先做本湖实验；WDC 等完成快照后按同一协议执行，不静默退回旧 WDC2k 或把新旧湖的结果混成主表。不额外启动数据扩建任务。

`taskA_protocol/inputs.json` 记录数据版本、构造策略、各 split 的 query/target/asset/qrel 数、source 分组、模型版本、特征签名、PCA 和索引来源。manifest 存在不等于构建完成，必须核对文件引用与统计。兼容的冻结 encoder 特征可复用；表内容、行视图、列顺序或模型设置改变时重新缓存。旧 Teacher 仅可作为初始化或迁移对照，不能直接充当新版数据已验证 Teacher。

重点审计：

- 隐式/显式 query、每 query 正目标和桥接属性数、正确列位置、每行可恢复属性、text/image 独有与共同支持的分布。
- 所有已知正目标完整进入候选及 positive mask；Q-E 排负使用所有正目标的有效 evidence 并集，E-T 排负也使用该 evidence 的全部已知正目标。
- 当前 `mining.py` 有多处仍依赖 designated evidence positive；须专项验证候选保留与 edge 负例构造，不能以此前数据层“多正例测试通过”代替 mining 链验证。
- 列名、普通 context、同源投影、实体重叠是否提供意外捷径；同源/等价 target 无法确认负例时跳过绝对 0 标签。
- 正 evidence 数量及来源不能系统性暴露标签；训练候选对正负目标使用相同构建规则，并保留真实在线检索的缺证据状态。

EntiTables 当前 manifest 指定每 query 5 行、构建期 `min_recovered_value_ratio=0.4`；现有 verifier 默认 `min_coverage=0.6`。A1 必须写清两者用途不同，不能把不足 3 行可恢复的 2/5 样本默认当作“系统本应能通过 60%”的失败。正式 Stage-2 主判据暂固定 60%，同时报告 40% 构建口径和可恢复上限；选择实验与 test 全程使用相同判据。

### A2. 划分与预算

- 使用新版 train/dev/test，按 source table 分组，相关 row views 不跨 split。
- 从 train 按 source group 留出 10% 为 `train_calibration`，其余为 `train_fit`；train_calibration 仅拟合诊断校准器或检查校准，不能同时用于 P/R 梯度训练。dev 用于模型与配置选择，test 在最终冻结方案后一次性评测。
- 固定搜索预算：Q-T 100、Q-text 20、Q-image 20、每 E-T 20；K={10,20,50} 从同一排名截断。此设置方便与旧 top-10 的预算对应，旧随 K 扩大检索深度的曲线另列 reference。
- 所有融合方法输出相同数量的唯一 targets；二阶段主预算 B=4 个 evidence/target。Oracle 注入只做瓶颈诊断，主结果不注入。
- 比較聚合/校准时共享同一模型产生的完整路径池；比较模型召回时各模型使用自身正确索引，不能只在旧模型候选上判定新模型的召回能力。
- 单关系核心比较做小规模 exact inner-product 复核。主要训练结论用 seeds 13/17/23；初筛 seed=13。

### A3. 评价指标与分母

| 指标 | 定义与用途 |
| --- | --- |
| Edge ranking | 四个 evidence 关系的 Recall@L、AUROC、AUPRC；分 random/hard/模态统计 |
| Edge confidence | 确认正负边的 p 分位数、Brier、NLL、可靠性图；报告标注集正例率与采样方式 |
| ValidPathRecall@K,B | 对有已验证恢复路径的 (Q,T) 正例，T 进入前 K 且其保留 B 条 evidence 中至少一条确实支持该 (Q,T) 的比例 |
| RowSupportCoverage@K,B | 对每个 implicit 正 (Q,T)，其保留 evidence 支持的不同 query 行数 / Q 总行数，再宏平均；T 未入选记 0 |
| RecoverableRowCoverage | 同上但分母为已标注可恢复行数；与全行口径同时报告 |
| Evidence-only discovery | T 不在同预算 direct 候选中、经 evidence 才进入最终前 K 的正确目标数和比例 |
| Fusion attribution | direct 前 K 未命中后被救回、原已命中后被挤出，分别计数；多正例按目标计并另报 query 级 |
| Retrieval quality | D-only、E-only、fused 的 Recall@K/MRR；implicit、explicit、模态、多正例分层 |
| Stage-2 quality | bridge-column accuracy、逐行值正确率、grounding、最终 join precision/recall、60%/40% 行覆盖成功率 |
| Cost | 每 query ANN、聚合、reader、定位、生成时间及完整延迟；索引大小、GPU 内存 |

旧 Coverage@10 保留但标为 legacy，不作唯一 evidence 判据。Path 指标根据 recovery-supported 路径计算，不能只看终点正确；缺少完备标注的 evidence 不自动判错，补做人工/独立标注的小样本精度审计。

多正例主监督比较集合概率 `-log sum_positive p(T)` 与 `-mean_positive log p(T)`。默认新版主配方采用后者以支持发现全部 targets，前者作为一次受控消融；历史 r5 复现保留它的原公式。正例集合指任务相关 target，局部 edge 的多正邻居不能互相成为负例。

### A4. 基线与瓶颈诊断

同一新版数据上重跑 raw、PCA-1024 identity、历史 r5 配方。历史 37.74/64.36 不再作为新数据门槛。另从 dev 按 source 分组固定最多 200 个 implicit query，作现有检索、gold E、gold T、gold T+E 的对照，记录每一步失败位置。

先修复并验证候选列位置分布。Oracle reader 超过 majority、position-only、schema-only 后才解释其语义能力；即使 reader 尚未通过，也可在 gold 列条件下做少量证据抽取实验来定位问题，不把全部 evidence 研究锁死在一个 gate 上。

产出：`inputs.json`、`label_audit.json`、`splits.json`、`baselines/`、`oracle_diagnostic/`、`RESULTS.md`。初筛优先使用可审计的缓存特征，不立即全量运行昂贵生成。

## 4. Task B：弱路径数量压倒质量的诊断与零训练聚合筛选

### B1. 先复现问题，并区分 raw 与 trained

分别对 raw、PCA identity、r5-repro 记录两跳边分数和 target 路径数量。弱/强标签以真实支持关系确定，不能把低 similarity 直接定义为无效。

原有路径 `g=s(Q,E)+s(E,T)`。以 raw 的示例说明：一条强路径两边各 0.75，g=1.5；四条弱路径两边各 0.5，LSE=1+log(4)=2.386，已超过强路径；16 条弱路径是 3.773。

即使未来 p 对无效边趋于 0，普通 `LSE(0,...,0)=log(N)` 仍奖励数量。全库 noisy-or 也不是自然解药：100 条仅 0.01 的路径给出 `1-0.99^100≈0.634`，相关或重复证据还违反独立性。只把分数压到 [0,1] 后继续无界累积，问题仍在。

### B2. 压力测试

固定 1/2 条有支持的强路径，增加 N={1,2,4,8,16,32,64} 条弱路径，分为：确认无效、同一 evidence 重复、相似正文/图片重复、支持同一行、支持不同新行。另做一条边强另一条弱、两条边都强但属性不一致的路径。

报告弱路径反超率、首次反超的 N、目标 rank 变化、分数与路径数的相关性，以及新增被正确支持行带来的收益。真实固定路径池和合成边分数曲线分开报告；合成结果只检验公式性质，不算真实任务增益。

正确预期不是“所有弱 evidence 永远无价值”：经验证且支持不同新行的较弱 evidence 可以有收益；无效或重复 evidence 应接近无增益。去重不移除合法的跨行信息，同一内容可以同时支持多行。

### B3. 先比较两跳组合，再比较跨路径聚合

第一组固定跨路径 max，比较：原始 s1+s2；置信度 min(p1,p2)；置信度 p1*p2。min 是瓶颈式默认，product 是软 conjunction 对照，不能仅因两个 p 看似校准就宣称乘积是联合概率。两跳属性一致性仍需 path 标签。

第二组固定同一组路径强度 q，比较下列聚合。默认先用 min；raw 先加单调校准器作为“只重标定、不训练表示”的对照，单列未经校准的原始系统。

| 编号 | 聚合 | 本轮用途 |
| --- | --- | --- |
| G0 | 原有 LSE，对全部路径 | 数量偏置基线 |
| G1 | max(q) | 不奖励数量的质量基线，可能丢失互补收益 |
| G2a | temperature-LSE，全量/真正 top-B | 给温度与截断提供匹配对照 |
| G2b | log-mean-exp，全量/真正 top-B | **优先实验、正式主候选**：去掉纯数量奖励 |
| G3 | 固定预算的 top-B 广义均值 | 有界累积，强调强路径；主候选 |
| G4 | G3 + 低置信度门槛 | 无效弱路径尽量不贡献；主候选 |
| G5 | 去重后的行支持聚合 | 直接体现 example rows 的互补性；必须做的故事主线对照 |

G2a 的 `temperature-LSE = t*log(sum(exp(q/t)))`，温度 t={0.1,0.3,1.0}；top-B 版本真正先取 top-B。G2b 使用相同温度、相同输入，减去 t*log(n)，n 为实际参与聚合的路径数。全量与 top-B 分开命名，不能把两种干预混为一项。

### B3.1. Log-mean-exp 专项：先做最小改动实验

历史核查：当前 `PATH_AGGREGATIONS`、r6/unification 扫描实现和可见 `work/` 实验配置没有 log-mean-exp；也未发现训练/检索聚合中 `LSE - log(n)` 的等价实现。过去的 top-k mean、softmax-weighted mean 和 edge z-score 都不是这个算子。因此这里列为新实验，不能引用旧 mean 类负结果排除它。

对路径分数 x，定义：

```text
LME_t(x_1,...,x_n) = t * log((1/n) * sum_i exp(x_i/t))
                  = t * logsumexp(x/t) - t * log(n)
```

优先顺序：

1. 在 raw 和 r5-repro 的已保存完整路径池上，**保持 x=s(Q,E)+s(E,T)、两跳组合、分数归一化和候选预算不变**，只把全量 LSE 换成全量 LME，先跑 t=1。这最直接回答当前数量奖励是否正在伤害 evidence 检索。
2. 固定同一输入，再比较 t={0.1,0.3,1.0} 的 LSE/LME 成对结果，分别做全量与 top-B。不要只报 LME 小温度相对 LSE 大温度的提升。
3. 在训练后的 q=min(p1,p2) 上重复；正式 Teacher/Student 交叉训练纳入 G2b，验证训练拉开边分数后是否仍需均值归一化。

t=1 的机制例子：

| 路径分数组合 | LSE | LME | 解释 |
| --- | ---: | ---: | --- |
| 一条强路径 1.5 | 1.500 | 1.500 | 单路径保持原分数 |
| 四条弱路径，各 1.0 | 2.386 | 1.000 | 不再靠数量超过强路径 |
| 十六条弱路径，各 1.0 | 3.773 | 1.000 | 同质量路径增多不提高分数 |
| 一条 1.5 + 四条 1.0 | 2.731 | 1.122 | 强证据会被混入的弱证据稀释 |

LME 落在参与分数的最小值与最大值之间；一组全部为 x 的路径，无论 n 都输出 x。这个性质直接符合“多条同质量弱路径不应仅凭数量压过少量强路径”。随着 t 变小，它更接近 max，减弱有限数量弱证据对强证据的稀释，但也减少对互补证据的累积。

需要单独测量的限制：

- 给一个已有强证据的目标加入大量弱证据，LME 可能下降。top-B 可限制尾部噪声，但不能笼统称 LME 对所有噪声鲁棒。
- 两条同强度 evidence 即使支持不同 query rows，也不会自动比一条得分高；互补性仍要靠 G5 或后续显式覆盖信号测量。
- 把整个路径集合复制相同次数，LME 不变；只复制其中一条强路径，会提高它的占比，所以 LME 不能替代内容去重。
- **若所有 target 参与的 n 都相同，同温度下 LME 与 LSE 只差同一个常数，evidence 排名完全相同。** top-B 且大家都有 B 条时尤其如此；若出现排名差异，先查输入、温度或实现。单独证据通道的 softmax/listwise loss 也不变；不同 target 的 n 不同时才改变它们的相对训练权重。跨通道绝对分数融合可能受这个常数影响，单独归因。

实现要求：n 只计 mask 内的实际路径，不计 padding；分数为 0 的合法路径仍计入 n。空集合由“无 evidence”状态处理，不计算 log(0)。同一固定路径集合上 LSE 与 LME 对各 x 的局部导数相同，不能把减 log(n) 表述为直接增强单条强边的梯度。证据分数进入 Stage-2 时保留同一 score-space 定义，不把 raw-LME 与 confidence-LME 混用。

正式配置候选名为 `logmeanexp`，支持独立 temperature 与显式 top-B 截断；该名字当前尚未实现。G2b 不因融合总 Recall 没升就提前取消，先看 evidence 排名、弱路径反超率和有效行支持。

### B3.2. 固定预算与可靠性门槛候选

G3/G4 公式如下，B=4，q 降序，少于 B 条补零：

```text
h_delta(q) = max(0, (q - delta) / (1 - delta))
S_power = (sum_{j=1..B} h_delta(q_(j))^r / B)^(1/r)
```

G3 使用 delta=0；G4 从 {0.25,0.5} 选择 delta，r={2,4,8}。阈值只在独立校准检查和 dev 选择中确定；数量受限的确认标签样本不支持可靠校准时，阈值仅称经验置信度阈值，明确其适用样本分布。

固定 B 而非有效 evidence 数作为分母，避免加入一条低质量 evidence 就稀释原有强路径。该式满足：零支持得零、范围 [0,1]、弱路径数量超过 B 后不再继续堆积、r 较大时更重视强路径。例：B=4、r=4，一条强度 0.8 得 0.566，四条强度 0.3 得 0.3。它并不保证任意四条中等证据必然输给一条强证据，这应由任务支持标签和独立性决定。

当前 `power_mean` 会先减去集合最小值，且分母为实际路径数；它不是这里的 G3，不能直接复用名字当作已实现。本轮新公式使用独立明确的配置值。现有 `logsumexp/top4` 也不等于真正 top-4 LSE。

G4 的硬门槛只用于路径贡献，局部 BCE/edge ranking 对门槛以下的边仍必须有梯度。实现对全零广义均值使用数值稳定且定义明确的零分支，验证训练梯度有限，不能靠 NaN 回退掩盖问题。

### B4. G5：把数量奖励改成行覆盖奖励

第一步基于现有 query-row embedding 的 row-E cosine，拟合按 evidence 模态的单调 row-support 置信度 w(i,E)，监督来源是 recovery 对应行；模型只看可见行与 evidence，不能使用 GT 行分配作为预测特征。

先在固定路径池上做一次质量+覆盖的贪心 B 条选择：每次加入使以下目标增量最大的 evidence，增量相同时按 q、ID 确定性打破平局：

```text
v(i,E) = w(i,E) * h_delta(q(Q,E,T))
S_rows(bundle) = mean_i max_{E in bundle} v(i,E)
```

同一行的重复支持收益饱和，不同行的支持可以相加；一条 evidence 可以支持多行。B、行数和筛选后的 evidence 小集合限定了成本，仍不做全库 bundle 枚举。空 bundle 得分为 0。

主对照在固定 q 排序的 top-L evidence 上选择 B 条，L=20，且 G2b/G3/G4 的此项覆盖对照也只消费这个相同的 top-L 输入池。全量 LME 的 B3.1 实验仍保留完整原始池，不能被这里的 top-L 替代。与单纯 top-B q 比较，记录 bundle 的 q 分布、实际行覆盖和终点质量。预测 w 错误时另用 gold w 作 Oracle 上界，不能把 Oracle 行混进主结果。

先保留 G5 为显式推理扩展。进入正式一致性训练时，候选 evidence 都由相同检索规则生成，在 batch 内按当前分数做同一选择；选择索引不反传，所选路径和 row-support 打分反传/冻结策略固定并记录。初版 w 冻结，只训练 Teacher/Student 的连接分数，避免同时改变太多模块。

预算：B1-B4 都是缓存分数/小型校准器实验；G2b 必须进入正式训练，另选 G3/G4 中一个主候选及 G5。最先执行 B3.1 的全量 LSE -> LME 配对，不能等待全部复杂模块完成才试这个最小改动。不因 G1 的单独最高 Recall 就跳过互补证据检验。

## 5. Task C：恢复 P 与 R 的受控学习

从同一 PCA-1024 初始化、R=I，使用相同训练列表、Teacher、目标和聚合。主控制使用同数据的 r5 path 目标与原始聚合，所有 checkpoint 另外按 A3 的 evidence 指标评测，避免先把 P 效应与新聚合混合。

| 臂 | P 学习率 | R 学习率 | 用途 |
| --- | ---: | ---: | --- |
| C0 | 0 | 0 | PCA identity；复用 A4 |
| C1 | 0 | 1e-5 | 固定 P、只训 R；同配方基线可复用 |
| C2a/C2b | 1e-6 / 1e-5 | 0 | P 单独贡献 |
| C3a/C3b | 1e-6 / 1e-5 | 1e-5 | 原始方案 P+R |
| C4 | 先 0 后选定 P lr | 1e-5 | 前 2 epoch R warm-up，随后 P+R |

batch=64，最多 12 epoch，最少完成 4 epoch 后按 evidence dev 主指标 patience=4；保留 epoch0/best/final。C4 至少完成解冻后 2 epoch 再判 early stopping。显式记录真实更新次数；不能把“epochs 相同但 step 不同”当作完全同预算。

P/R anchor 默认各 0.1，输入维度与矩阵大小归一化；实现独立系数以便归因，但主扫不额外扫 anchor。只有 P 更新接近零且梯度/anchor 诊断支持过强约束，才补一次 mu_P=0.01 对照；不按未经验证的漂移阈值直接认定原因。

记录各类型 P 漂移、各关系 R 漂移、梯度与参数范数、score 分布和四种局部检索指标。P 改动后同步重算所有目标向量并构建匹配 ANN；old-index 结果无效。C2-C4 必须完整报告，不能仅因 direct 暂时下降就停止整个方向。

Teacher 的基线来源在 A 固定；新数据需要同口径训练或明确标为旧 Teacher transfer。C 组不在不同 P/R 臂间切 Teacher。C1 超过 C3 时也保留最佳 C3 进入 D/E，以免在未修复局部边监督时提前否定 P。

## 6. Task D：绝对连接性监督、局部边训练与一轮 mining

### D1. 标签与概率学习

对确认标签训练 `BCEWithLogits(ell,y)`，y 为局部边有效性。每种关系先在列表内部平均，再对非空关系宏平均，避免 text 或样本多的关系吞没 image。负例由确认无效、人工审核的难负例或可验证的 corrupted path 组成；无法确认的 ANN 非 qrel 仅进入原有弱监督 ranking，并单独计数，不进入高置信度 0 的 BCE。

初始采样每条正边配最多 4 条确认负边，其中 random/hard 各半；不足则不伪造标签。此采样的 p 不自动等于全库先验下概率。校准诊断必须说明来自训练配比还是实际检索候选分布，后者以独立固定的标注样本评估，并报告样本不足的关系。

训练目标保留局部排序和 target-level 监督，默认所有 list loss 用未截断 logit 计算：

```text
L = L_path_D + L_path_E
    + lambda_local * (L_edge_rank + lambda_abs * L_edge_BCE)
    + lambda_KD * (L_KD_edge + L_KD_path_D + L_KD_path_E)
    + mu_P * Anchor(P) + mu_R * Anchor(R)
```

默认 lambda_local=1、lambda_abs=1、lambda_KD=0.3、KD temperature=1。缺少确认标签/有效列表的项不计，但记录实际参与率和分关系 loss；不能把未被训练的关系当作已训练对照。p 的仿射参数在训练和推理使用同一 checkpoint，不做 query 内 min-max 伪装概率校准。

有界聚合输出 S 若直接进入 target softmax，分差最多 1，可能不足以优化宽列表。对 confidence 输入的 G2b/G3-G5 定义 `evidence_target_logit = S / t_target`，默认 t_target=0.1，所有同池对照使用相同规则；只在 dev 明显饱和或梯度不足时补 {0.05,0.2}。t_target 与 p 的边校准温度、LSE/LME 温度、KD 温度分别命名，不能共用一个 tau。

### D2. Teacher 同样接受局部连接性检查

在新数据的 train_fit 上训练 Teacher edge，再做 path；保留 cross-object Relation Transformer，不扩大架构。记录 edge ranking、绝对边置信度与有效路径能力，不能仅凭 Q-T rerank 判断 Teacher 是否适用于 image/text 边。

主实验先用纯 Teacher KD；Teacher+raw 混合只作历史/抗噪对照。若某关系 Teacher 弱于 supervised Student，保留该关系的无 KD 臂并解释结果，不删除模态。最终有/无 KD 对照必须使用相同列表、P/R、训练步数及聚合。

为了识别 Teacher 细粒度贡献，额外训练一个低成本 pooled-object pair MLP 对照，在相同候选池测边与路径质量；它只作为诊断，不替换主 Teacher。若无法支持细粒度优势，如实限制论文表述。

### D3. 初筛训练臂

使用 C 中固定的一套 P+R 设置，主聚合暂保持原始路径和 LSE，隔离局部训练的效果：

| 臂 | 局部训练 | path 阶段 | 要隔离的因素 |
| --- | --- | --- | --- |
| D0 | 无 warm-up | 原 path-only | 对照，复用 C3 |
| D1 | 2 epoch edge ranking+KD | path | 恢复 Stage C1 |
| D2 | 与 D1 相同 | path + 持续 edge ranking/KD | 局部关系遗忘 |
| D3 | 同 D2，加 edge BCE | path + 持续 edge BCE | 学习绝对连接置信度 |

固定总更新预算为 C3 的最多 12 epoch 等价步数；warm-up 占前 2 epoch 等价步数，剩余用于 path。D2/D3 每个 path batch 对应一个 edge batch，记录算力增加；另用等 wall-time 曲线核查额外算力是否解释收益。所有初筛先 seed=13。

比较 raw/calibration-only/D2/D3 的分数直方图及 AUROC/AUPRC。仅 histogram 拉开而 ranking、有效路径没有改善，不能称学会连接性；只在随机负例上区分得好，也不能宣称解决伪桥接。

### D4. 必须做一轮健康 Student mining

以 D2/D3 中 evidence 表现更好者为源，在 train_fit 上挖 Q-E、E-T、完整 path 的高分错误，外加 direct hard target；独立配额起点：每模态 Q-E 8、每正 E 的 E-T 8、每 Q 完整 path 8、direct target 8。原有 raw-ANN lists 保留，因此这是动态负例更新的对照。

高分错误路径不自动代表其 Q-E 或 E-T 局部边为负；先判断错误在哪一跳或属性一致性，再分配对应监督。确认缺失时仅标注路径负例，不把所有边一并压到 0。在三类 evidence 难例中按关系抽查，汇报已确认/未确认比例。

所有多正例和等价候选排负检查通过后，用 25% 新难例 + 75% 原列表继续 2 epoch，对照组从同一 checkpoint 用 100% 原列表继续相同步数；两组 P/R lr 相同，不让新增 hard-data 默认改变学习率。结果记为 D4-mined 与 D4-control。mining 不再依赖旧 WDC direct gate 或某个整数 Recall 门槛。

## 7. Task E：训练与 aggregation 的交叉实验

这是本轮必须完成的核心机制表，不允许只拿各自最优点拼起来。

先固定同一 P+R 初始化、训练列表与 hard-negative 池、Teacher 架构和训练步数，使用相同的 p 变换与两跳 min 组合。T0 的所有网络参数都可接受 ranking/path/KD 梯度，T1 只增加绝对 edge BCE：

|  | G0：全部 q 的 LSE | G2b：log-mean-exp | G*：B 选出的 G3/G4 |
| --- | --- | --- | --- |
| T0：局部 ranking+KD，无 BCE | E00 | E01：只改 LME | E02：只改跨路径筛选/聚合 |
| T1：T0 + edge BCE | E10：只加绝对边监督 | E11：训练 + LME | E12：训练 + G* |

这张表里的聚合输入都是同一 q=min(p1,p2)，确保只改变跨路径算子；原始 LSE(s1+s2) 系统另外列作 legacy，不混称为这一因素的单变量对照。E 中所有单元格的 target-logit 温度同为 0.1。G0 与 G2b 正式主效应比较使用相同的全量路径和相同 LSE/LME 温度；top-B LME 最优时另列 matched top-B LSE 对照，不能把截断或温度变化归因于减 log(n)。

Teacher 与 Student 都针对每个单元格训练匹配的目标、p、两跳组合和聚合，重新生成对应 logits。E 中 T 表示整个 Teacher/Student 链的绝对监督设置；它不能单独归因于 Student，因此另做同一选定 Teacher 下 Student BCE 开/关，以及同一选定设置下 KD 开/关两个对照。训练列表若 n 完全相同且数学上目标等价，可以复用该训练链，但必须证明目标等价并注明，不能伪装成两次独立实验。

报告训练主效应、LME 主效应及交互 `E11-E10-E01+E00`，另报 G* 的交互 `E12-E10-E02+E00`，依据逐 query/target 指标计算；同时呈现弱路径反超曲线，不能只列一个最大 Recall。B 的所有推理重组仍保留为零训练诊断，正式行使用 train/inference 匹配配置。

最后增加 E13：T1 + G5 行覆盖聚合，并做 matched Teacher/Student 训练，比较 G2b/G* 与 G5。G5 用相同的 L=20 输入池、B=4 输出预算、冻结 w，覆盖变化不能来自更多 evidence。

如果最终候选相对固定 P 尚未受控比较，补一条同设置 P-frozen 对照，完整隔离 P 对最终新目标的作用。多正例新/旧 loss 在最终候选上补一条对照，不从历史不同数据集结果归因。

## 8. Task F：允许 evidence 发挥作用的 fusion

先锁定 E 的最多两个 evidence 候选，在同一检索池比较：

| 配置 | 规则 | 地位 |
| --- | --- | --- |
| F0 | direct-only | 必须的质量与成本基线 |
| F1 | evidence-only | 桥接检索能力 |
| F2 | weighted RRF，D=1/E=0.05 | 历史工程对照 |
| F3 | 等权 RRF，D=1/E=1，k=60 | 无新训练的主控制 |
| F4 | D/E 并集补算 direct 分数，再按校准分数融合 | 主候选 |
| F5 | 固定 K 内 D/E 各预留一半，去重后从两侧交替补齐 | 候选准入瓶颈诊断 |

F4 对并集中的每个 T 补算 s(Q,T)，evidence 无有效路径时支持为 0。起点为 `score=(1-lambda)*p_D+lambda*S_E`，lambda={0.25,0.5,0.75}，仅在 dev 选；p_D 与 S_E 均为 [0,1] 但含义不同，必须检查分布与实际效用，不因同范围就宣称已校准。

主轮不默认增加复杂自适应 gate。若固定融合仍有清楚的条件差异，再把置信度、行覆盖、一致性做成简单单调规则的小扩展；不使用 GT 属性/模态需求在在线时选择分支。真实 evidence 删除/替换后无任何变化，应报告未建立证据依赖。

最终模型选择看 ValidPathRecall、RowSupportCoverage 及 Stage-2 的实际收益；evidence 权重较高本身不算成功。direct 已发现 T、但必须 evidence 才能恢复合法连接值，同样计入主线收益，并与 evidence 独立发现的 T 分开。

## 9. Task G：小规模二阶段闭环与证据干预

沿用冻结 Qwen3.5 reader 和线性列头先做可识别性实验，修复列位置偏置。reader 训练只用 train；dev/test 行和 target 不泄漏。再按 A4 固定样本评估以下四组：

1. 真实 evidence。
2. 无 evidence，但 Q/T/候选列和推理预算一致。
3. 同实体、不支持当前属性的 evidence；若能支持其他目标不将其当作全局无关。
4. 属性相似但实体错的 evidence，或 image 必要样本上的关键区域遮挡。

报告列准确率、值正确率、引用支持和最终 join，而非仅生成值在目标列中出现。无 evidence 时应允许拒答；不能给另一组额外提示或缩小候选集。

对默认 B=4、5 行 query，当前唯一行分配使最多 4 行获得 evidence，且一条 evidence 服务多个实体时会丢信息。用同一 bundle 比较 argmax 唯一行与每 evidence 至多 top-2 行并带拒绝阈值的稀疏路由；每 query 总 localization 预算固定为 8，实际调用数与延迟一并报告。w 的阈值在 train_calibration/dev 固定，不能测试时挑 GT 行。

每行单 evidence 与至多两条互补 evidence 的生成做对照，输入总 token/image 预算相同。定位先比较全文/全图与 FOCUS crop/span，在最终值正确性上证明裁剪是否有用，而非只展示 heatmap。

主闭环按全局 top-1 表-列完成一条桥接；多正例另比较 top-2 表-列，限制总 reader/localization/generation 预算，并分别报告“任一成功 join”和“全部正目标 recall”。增加 no-valid-column 或负表级兼容训练作为条件扩展，只有观察到错误表无法拒绝时实施。

如果小规模真实证据无收益但 gold 值可使 join 成立，明确瓶颈在定位/补值；如果 gold evidence 的有效行不足 60%，明确数据上限。任何一个模块未通过都要产出阶段结果，不能用另一个模块的 Recall 替代或编写“通过”结论。

## 10. 模型选择、统计与实验预算

### 10.1 选择顺序

训练期 primary 为 implicit `ValidPathRecall@10,4`，tie-break 为 `RowSupportCoverage@10,4`、implicit target recall、总体 recall、较早 epoch。量很小/大量并列时同时保留最多两个 evidence/总体质量的非支配 checkpoint，不用 0.05 fused gate 筛掉 evidence 模型。

最终在 dev 的 G 阶段以 implicit 60% 正确值覆盖下的 bridge success 为首要指标，同时报告 join precision 与总体 recall。与同数据 direct/r5-repro 相比总体 R@10 超过 2 个百分点的下降记为 tradeoff，仍可保留为机制结果；主表不隐藏退化。需要部署配置时从满足质量容忍范围的候选选取，不把容忍不劣当作显著提升。

test 不再选择聚合、阈值、融合权重或 checkpoint。按 source group 做 paired bootstrap（10,000 次，seed=13），因为同源多行视图、多正目标不独立；同时报告三训练 seed 的均值/标准差，bootstrap 不能代替训练随机性。报告分层样本数，尤其 image 独有和 WDC 小桶。

无论成功或失败都生成 FINAL.md，分别给出“实现/数据有效性”“主线证据”“总体性能”“下一步”。不能因门槛未过而不生成正式结论。

### 10.2 预算与推进

| 阶段 | 初筛新增训练预算/湖 | 说明 |
| --- | ---: | --- |
| A | 1 个新版 Teacher 链 + 1 个 r5-repro Student | raw/PCA 无训练；Teacher 可兼容复用但必须记来源 |
| B | 0 个神经主训练 | 校准器与缓存分数扫描单计 |
| C | 最多 6 个 Student 链 | C1 可复用 A；C2/C3 各 2、C4 1；含最多 1 个 anchor 条件补扫 |
| D | 3 个 Student 链 + 2 个两 epoch continuation | D1-D3，D4 control/mined；确认负标签不足先补小样本审计 |
| E | 7 个匹配 Teacher/Student 单元格 + 最多 4 个 Student 对照 | E00/E01/E02/E10/E11/E12/E13；Student BCE、KD、P、正例 loss 对照 |
| G | 1 个 reader 线性头；条件扩展单独计 | 先最多 200 个 dev query，不全量生成所有配置 |

初筛上限为每湖 23 个 Student 链或 continuation（A1+C6+D5+E11），最多 8 个主 Teacher 链（A1+E7），另 1 个小型 pooled-pair Teacher 诊断。Teacher/Student 的 edge+path 视为一条链，训练步数及 GPU 时另记；数学等价的 LSE/LME 训练可在验证后复用以减少成本。此上限不包含条件的 matched top-B LSE 新训练；若需追加，最多 2 个匹配链/湖并单列成本。G2b 的第一轮零训练筛选不增加主模型训练预算。

初筛后只对最终候选及其主对照补 seed 17/23，最多额外 4 条 Student 完整链/湖；这组估计条件于固定 Teacher。若要宣称 Teacher-Student 整链稳定，再为两个额外 seed 重训 Teacher 并单列成本，不把固定 Teacher 的 seed 结果冒充整链方差。

每阶段先小样本 smoke，预计缓存/GPU 成本落盘后再跑该阶段正式臂；不启动全超参笛卡尔积。WDC 若尚未完成数据快照，则标 pending-input，仅推进已可执行的 EntiTables。运行时根据实际空闲 GPU 调度，不能继承旧脚本自动占用两张卡或影响数据构建任务。

执行依赖：A -> B 与 C；C -> D；B+D -> E；E -> F；A 的 Oracle 诊断及 reader 可先进行，F 候选固定后 -> G -> seeds/test/FINAL。任何阶段的机制负结果都留存，不删除未达标 checkpoint 的唯一证据。

## 11. 实现清单与正确性检查

| 模块 | 必要改动 |
| --- | --- |
| `src/mmdd_stage1/models.py` | 正尺度的 type-pair affine confidence；P/R 独立学习率/anchor 与漂移沿用现有实现 |
| `data.py`、`construction.py`、`mining.py` | confirmed/unknown 边标签、所有正邻居、完整多 target 候选、严格 row-attribute support；训练可见数据不含 GT provenance |
| `objectives.py`、`training.py`、`scoring.py` | 优先实现 logmeanexp 及 masked count/temperature；多正例目标、edge BCE+ranking/KD、持续 edge/path 训练、明确两跳组合、新 G3/G4/G5、target 温度 |
| `retrieval.py`、`evaluation.py` | 同公式的训练/检索聚合、固定候选预算、D/E 并集补算、有效路径/行覆盖与分层指标 |
| `teacher_logits.py`、`checkpoints.py` | 缓存签名覆盖 edge transform、两跳组合、聚合、门槛、B/r/温度与 row-support 版本；旧缓存不冒充新打分 |
| `src/mmdd_stage2/` | 去位置偏置检查、证据反事实、稀疏路由、小规模值与 grounding 评测 |
| `src/` 入口与 `configs/` | 新 r10 入口和显式配置；不从旧 r9 continue 脚本调度，支持明确 split=train/dev/test |

正确性测试针对行为，不为文件存在或参数转发堆测试：

- 单调 confidence 不改变单关系 ANN 排序；P/R 冻结臂确实不更新相应参数，R lr=0 不被默认值覆盖。
- LME 同值集合与单值返回该值、完整复制集合不变、分母不含 padding、0 分合法路径仍计数；固定 n 的 LSE/LME 排序与局部梯度等价；全零/空集行为正确。
- G3/G4 全零及空 evidence 返回 0，padding 不贡献，预算外弱路径不继续加分，permutation 不变，CPU/训练张量版与检索版一致，梯度有限。
- 精确重复的 asset/content 不重复计票；多条不同证据支持同一行趋于饱和，支持新行则提高覆盖；单 evidence 可以支持多行。
- “强+弱”的两跳不能由强边完全掩盖弱边；不能把 q 当成独立概率。
- 多正例保留与排负覆盖 construction -> mining -> edge/path loss -> evaluation；未知负例不进入 confirmed-negative BCE。
- 有界 S 的 target 温度不修改局部 ANN score；edge/KD/aggregation/target 温度分别影响正确组件。
- F4 evidence 独有目标可进入最终 top-K；总目标/evidence/生成预算受控。
- test 数据无法用于 training/mining/calibration；语义正确但列位置变化不应系统性破坏 reader，错误填值不能仅凭 target containment 被计为正确恢复。

相关 pytest 使用 `MMDD` 环境，从隔离临时工作目录运行并使用合成配置；不允许任何测试隐式加载用户 `.env.openai`。本轮主训练使用本地模型，不要求 OpenAI 客户端。需要额外模型标注时另行明确数据和调用范围，不触碰或查看秘密文件。

## 12. 必须交付的实验图表

1. raw / 仅校准 / P+R ranking / P+R+BCE 的边分数分布、排序质量与可靠性图。
2. 固定强路径下弱路径 N 的反超曲线，另列重复证据与真实跨行支持。
3. T0/T1 × G0/G2b/G* 的 2×3 主效应及交互表，外加 G5 行覆盖结果；LME 与 LSE 同温度/同路径池的对照必须出现。
4. P、局部 edge、BCE、mining、KD、多正例 loss 的同设置消融。
5. direct-only / evidence-only / 0.05 RRF / 等权 RRF / 最终 fusion 的准入、救回、丢失与最终补值贡献。
6. 真实/移除/替换 evidence 的逐行值恢复与 join precision/recall，implicit 和 explicit 分开。
7. 每湖完整运行 manifest、逐 query 指标、seed/source-group CI、成本及失败样例。

本轮成功的科学证据应是：训练让真正无效边的置信度下降并改善难例区分；聚合减少无效数量优势、保留独立行支持的收益；两者结合后，在相同预算下提升可用证据与正确属性恢复，最终带来真实 joins。若只能提高 direct 召回或改变分数外观，应如实记为局部结果。

依据：[前序审查](docs/stage1_stage2_story_audit_20260907.zh-CN.md)、[r6 聚合实验](work/stage1_optimization_r6_20260830/taskD_path_aggregation/entitables_RESULTS.md)、[Stage-2 round1](work/stage2_round1_20260831/RESULTS.md)。校准与排序的区分参考 [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)，listwise distillation 参考 [RocketQAv2](https://aclanthology.org/2021.emnlp-main.224/)。G3-G5 与完整实验设计是本项目待验证方案，不借引用宣称它们已经有效。
