# Stage-1 分湖 Teacher 数据生成审计(2026-08-29)

## 目标

排查 r4 中 WDC-only teacher 重排大幅回退(−23.27%)是否由**数据生成或代码改造引入的 bug** 导致,而非单纯的样本不足。本任务只审计、不训练、不改变任何现有行为,产出结论与修复建议清单。

## 背景

- 分湖后,各湖是各自独立的 teacher + student 实例(框架共享、实例按湖)。
- 关键对比事实(问题信号):
  - **混合训练的旧 teacher(task7 cache24k)在 WDC 上重排 −13.9%**;
  - **WDC-only 重训 teacher(taskM/per_lake/wdc)在 WDC 上重排 −23.27%**。
  - 同样是 teacher,同样的结构,小湖自己训的反而明显更差 → 疑似数据生成或拆分问题,而不只是样本不足。
- 发现的**明确不一致**(FINAL.md 中可证):WDC 的 student 是用 Task-M 的 WDC-only teacher 蒸馏的,但 WDC 在线重排用的 reranker 是 Task-J 的混合 teacher——蒸馏源与重排器是**两个不同的 teacher**。审计需一并确认这是否为有意设计(如果是,说明问题只是重排器未升级;如果不是,是管线不一致,需修复)。

## 审计项(按怀疑优先级排序)

### 1. 分湖语料拆分的整洁性(最高优先)

**怀疑**:evidence 资产(text/image)在分湖时可能被错误归属——例如同一资产同时是两个湖的证据,被重复计入两边或只计入一边,导致某湖 teacher 的证据列表被污染或缺失。

**要查证的**:
- `stage1_corpus.jsonl` 拆成 `entitables_corpus` / `wdc_corpus` 的拆分脚本逻辑:对象 ID 归属如何判定?是按"对象所属源数据集"还是按"该对象在哪个湖被引用"?
- **证据资产的归属**:每个 evidence asset(text/image)在拆分后只出现于一个湖,还是可能同时出现在两个?证据资产是否被重复?(若同一 asset_id 在两个 corpus 中都存在,违反唯一性;若 evidence 被算作 query 或 target 的一部分,则归属规则需说明。)
- 拆分量校验:两湖对象数之和 == 混合语料对象数?两湖对象 ID 集合无交集?分湖后 evidence 数量与混合时对每个湖实际使用到的 evidence 数量一致?
- 特别确认:**query_tables 是否按湖正确归属**(一个 query 表只属于一个湖,不能因被另一个湖的 target 引用而跑到另一个湖)。

**产出**:拆分前后对象计数表(table/text/image × 每湖 × 混合),明确列出任何重复或归属可疑的资产 ID。

### 2. WDC-only teacher 训练列表的重建(次高优先)

**怀疑**:taskM 为 WDC 重建的 edge/path 训练列表,其**负样本来源**和**证据绑定**可能与混合版不一致,导致 teacher 学到的分布与重排任务错位。

**要查证的**:
- WDC 训练列表的负样本是否只来自**本湖** raw ANN?重建时是否误用了混合语料的 raw ANN(带跨湖干扰项)?
- WDC teacher 的 edge 列表宽度、handcrafted negatives 配比、evidence 绑定是否与 EntiTables 用同一套生成逻辑(对比两个湖的生成日志/配置)。
- WDC path 列表的 candidate evidence 集合是否被正确重建——分湖后 `recovery_evidence`/`evidence_by_target` 是否按湖重算,还是沿用了混合时的全局版本?
- **关键换算**:WDC 分湖后的 path 训练列表数量(3,766)比混合时 WDC 部分(3,564)更多还是更少?多出的来源是否合理(如重建时把原本丢弃的 evidence 补进来了)?若数量异常膨胀或缩水,查明原因。

**产出**:两个湖训练列表的生成配置逐项对比表;WDC 列表的负样本抽样日志抽查几条,展示其检索来源。

### 3. 蒸馏源与重排器不一致(明确问题,评估影响)

**要查证的**:
- WDC 在线重排(`taskL_end_to_end`,`per_lake/wdc`)指定的是哪个 teacher checkpoint?(应为混合 teacher `1080b44c…`。)
- WDC student 的 KD 蒸馏目标(`taskM/per_lake/wdc/student_kd0.3`)用的是哪个 teacher 的 logits?(应为 WDC-only teacher `bf6c7646…`。)
- 若确实不一致:WDC 管线的"蒸馏源 teacher"与"在线重排 teacher"不同。评估:在线重排若切成 WDC-only teacher(WDC 重排 −23.27 的那个)是否更差?当前现状(混合 teacher 重排)是否反而更优?——即**确认这是否是"用混合 teacher 重排"的隐性正确选择**。

**产出**:两处 teacher 的 SHA-256 与来源目录;结论"不一致是设计还是疏漏"。

### 4. in-batch negatives 在 WDC 的退化(次优先)

**怀疑**:WDC 湖内候选对象少,in-batch negative 池可能太小,导致 WDC-only 训练时负样本多样性不足,teacher 学不到判别力。

**要查证的**:
- WDC 训练时每次 in-batch 的实际 negative 数量分布(对比 EntiTables)。WDC 表对象数和 text/image 证据对象数各有多少?
- `in-batch-max-negatives`(256)在 WDC 湖内是否实际达不到(即池子本身就 < 256)?若达不到,负样本是否被重复使用?

**产出**:两湖 in-batch negative 数量直方图/统计;若 WDC 池 < 256,量化差距。

### 5. 统计功率说明(附带)

**要查证的**:
- WDC dev 202 查询、单正例,WDC-only teacher 重排的 −23.27% 的相对方差(用配对 bootstrap 重算该数的 CI)。
- 若 CI 极宽,把"样本不足导致的方差"与"真实回退"量级分开,判定 −23.27% 是否在噪声范围内有可靠信号。

**产出**:WDC teacher 重排 delta 的配对 bootstrap 95% CI;对"该回退是否统计可信"给明确结论。

## 判定规则(输出三种结局)

- **A. 存在数据生成 bug**(如跨湖 evidence 污染、负样本用错语料、蒸馏/重排 teacher 错配)→ 列出 bug 位置与修复动作,标记"修复后可重跑 WDC teacher";这是优先要抓的。
- **B. 数据生成干净,回退来自样本不足/统计方差** → 明确结论,WDC teacher 维持现状,修复建议转为"补数据"(负样本可从混合语料取、检索/评测仍在本湖)或"WDC 用混合 teacher 重排"。
- **C. 数据生成干净,回退来自模型结构/WDC 湖类型不匹配** → 支持"teacher 在表面相似主导湖自动退化"的叙事,WDC teacher 不重训,直接写进论文。

## 产出物

- `work/stage1_optimization_r4_20260829/task_audit_wdc_teacher/audit_report.md`:逐审计项的发现、证据(引用具体文件/日志/ID)、结论(ABC 归属)、建议;附必要的对比表与 ID 列表。
- 可复现的审计脚本(如需读语料/训练列表/checkpoint 的,写成一个可重跑的脚本,一次性输出所有统计到 JSON + markdown)。
- 零训练,不修改任何现有文件与行为。
