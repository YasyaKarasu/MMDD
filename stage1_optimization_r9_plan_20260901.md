# Stage-1 第九轮计划:解冻投影 P + student-edge KD 回归 + hard-negative mining 解冻(2026-09-01)

## 背景:r8 与 pipeline unification(work/stage1_pipeline_unification_20260831)已确认的结论

1. **当前论文 primary 仍是 r5 配方**:EntiTables fused R@10 = **37.74%**(τ=1.0,+重排 46.16%),WDC = **64.36%**(τ=0.7,重排 no-op)。r8 曾采纳的 WDC 65.35% 已被 unification 轮认定为"训练用旧聚合、推理用新聚合"的**不一致配置,reference-only,不可采用**(`work/stage1_pipeline_unification_20260831/FINAL.md`)。**本轮所有 gate 锚点 = Enti 37.74 / WDC 64.36。**
2. **结构杠杆已用尽,但用尽的全部是"约束 R"方向**:anchor(减速不改向,R2)、按湖实例化(R4)、低秩残差(R7 证伪"坏方向太多")、KD 温度(R7 关闭)。**从 R1 健康配方确立至今,以下三个自由度从未在健康基座上测过**:投影 P(全程 `--freeze-projection`)、student-edge 阶段(R1 结论后跳过)、ANN 挖掘负样本(R1 起冻结,所有 run `hard_examples: 0`)。
3. **student-edge 被砍的证据已过时一半**:R1 的归因是"in-batch edge 伤 7pt、旧 Teacher edge-KD 毁 26pt"——那是混合语料 + 有毒 Teacher + 旧口径下的结论。现在有分湖实例化、被 R5 证明有锚定作用的 KD 形式 `τ·z(s_T)+(1−τ)·z(cos)`、以及修好的路径聚合。
4. **mining 的解冻前提早已满足**:R1 冻结 mining 的理由是 (a) student ANN 索引已崩坏、(b) Teacher 打分不可信。(a) 自 R4 起解除(两湖 student 均显著超 raw);(b) 对 EntiTables 解除(Task-M teacher gate pass),对 WDC 部分解除(teacher 重排仍是负资产,但 R5 证明蒸馏形式能过滤 teacher 噪声)。R2 FINAL 原话"Hard-negative mining remains frozen: the prerequisite … is not met"——该前提现已 met,一直没人回头。
5. **evidence 通道是当前最明确的差口**:Enti evidence R@10 = 9.81%、coverage@10 = 3.41%;r8 Task R2 的最优候选 fused 37.95% 就差 evidence 0.19pt 没过"≥10%"双赢门禁。
6. **扩数据湖本轮不做**(用户决定)。r8 Task R1 已记录当前 WDC 池为 `intermediate_current_pool`(深检索仍有余量,P95 正例排名 762);该事实保留,扩池实验不进入本轮范围。

### 本轮要回答的核心问题

**在健康配方(r5)基座上,三个从未动过的自由度——P、student-edge、mined negatives——各自能否带来 CI 支持的增益?若单项能,联合后是否仍能(R6 教训:贪心 ≠ 联合)?若 P 解冻(线性)不够,非线性容量(残差 MLP)是否是下一级杠杆?**

## 全局约束

- 评测口径不变:`recall_ks={10..50}`、γ=10、γ_e=2、配对 bootstrap(10,000, seed 13)、按湖 CI gate(tolerance 0.02)、epoch-0 留存 gate、同池对比(HNSW 独立重建的近似不确定性口径沿用 r7 结论,关键对比优先用共享候选池或穷举复核)。
- **基座配方 = r5,模板脚本 = `work/stage1_optimization_r5_20260829/run_task1_tau_ablation.sh`**。所有训练 arm 除本任务声明的干预外,逐 flag 与模板一致(含 τ:Enti `--kd-target-teacher-alpha 1.0` / WDC `0.7`,`--distillation-weight 0.3`,μ=0.1,in-batch 256,epochs 12,batch 64,patience 4,seed 13)。输入路径(features / corpus / raw_index / pca / teacher)一律取模板脚本 case 分支中的定义。
- **⚠ 已知的默认值陷阱(每个 arm 必须显式规避)**:`train_stage1.py:1359` 中 relation lr 的默认是 `1e-5 if frozen else learning_rate`——一旦去掉 `--freeze-projection`,R 的学习率会静默从 1e-5 跳到 1e-4。**本轮所有 arm 无条件显式传 `--relation-learning-rate`。**
- **训练/推理聚合一致**(unification 教训):所有训练与 primary 评测使用 `logsumexp + evidence-top-k 4 + weighted_rrf e0.05`(两湖同)。r8 式检索侧重组(WDC `topk_sum`)只允许作为 reference 行报告,不参与 gate。
- Teacher 不动(Task-M 两枚);语料/候选池不动;image 模态保留;WDC teacher 在线重排维持 no-op。
- 对 WDC 的诚实预期:历史上每多一个自由度 WDC 就回退一次(共享模型、纯监督、低秩 μ=0、混合负样本 Teacher)。**主战场是 EntiTables;WDC 每 arm 独立 gate,任何 arm 允许按湖回退保持 r5——分湖差异化配置是既定实践。**WDC 结论只在 CI 口径下陈述(202 查询,1pt=2 条)。
- 结果写入 `work/stage1_optimization_r9_20260901/`,逐任务 RESULTS.md + 轮末 FINAL.md;新增代码一律配测试;checkpoint 只保留 epoch-0/best/final。

---

## Task A0:投影漂移插桩(前置,零训练)

**理由**:解冻 P 后若不逐 epoch 记录 P 的漂移,Task A 的失败模式(P 没动 vs P 动了但没用 vs P 发散)无法区分,Task D 的触发条件也无从判定——等价于 R2 给 relation 加 `relation_drift` 落盘的动作。

**实现**:
1. 在 student 训练 history 记录中新增 `projection_drift`:每 epoch、每 object type 记 `‖P_t − P_pca‖_F / sqrt(4096·1024)`(与 anchor 的归一化一致);冻结时恒为 0,向后兼容。
2. 确认(已核实,记录进 RESULTS 即可):解冻时 anchor loss 自动含投影项 `‖P−P_pca‖²/(4096·1024)`、与 R 共用 μ(`src/mmdd_stage1/training.py:359-368`);投影参数组的 lr 由 `--learning-rate` 提供(`src/train_stage1.py:219-241`)。
3. 测试:冻结 run 漂移恒 0;解冻 run 漂移单调可记录;history JSON 字段齐全。

**验收**:测试通过;一次 1-epoch smoke run 的 history 含逐 type 漂移。

---

## Task A:解冻投影 P(本轮最高优先的训练实验)

**理由**:P 是自 R1 起从未在健康基座上训过的最大自由度(唯一训过 P 的 initialization A/B 跑在有毒 Teacher 基座上,按 R1 自己的方法论属无效消融)。三个独立的收益假设:(i) 追回 PCA 压缩代价(零训练探针:PCA-1024 只保留 raw R@10 的 95.7%);(ii) 把子空间转向任务相关方向(PCA 是无监督选的);(iii) **三个 object type 的投影解冻后可以分化**(冻结时共用同一 PCA 基,`models.py:495`)——这是全新的表达能力。

**实现**(每 arm = 模板脚本去掉 `--freeze-projection`,再按下表覆盖):

| arm | 干预 | `--learning-rate`(=投影 lr) | `--relation-learning-rate` | runs |
| --- | --- | --- | --- | --- |
| A1 只训 P | R 冻结,隔离 P 的贡献 | {1e-5, 1e-6} | **0**(AdamW 组 lr=0,含 weight-decay 均不更新) | 2 湖 × 2 = 4 |
| A2 P+R 联训 | 完整解冻 | {1e-5, 1e-6} | **1e-5**(显式钉住,规避默认值陷阱) | 2 湖 × 2 = 4 |

- 其余全部同 r5:τ(Enti 1.0 / WDC 0.7)、KD 0.3、μ=0.1、in-batch 256、epochs 12、patience 4。
- 逐 epoch 记录 `projection_drift`(Task A0)与 `relation_drift`;评测沿用模板的逐 epoch dev 检索 + `evaluate_stage1_r3_baselines.py` 终评。
- **A3(条件子扫,仅当触发)**:若 A1+A2 全部 8 组的投影漂移终值 < 1e-3(μ=0.1 把 P 压死)→ 加 `--projection-anchor-weight` 解耦 flag(代码+测试),对每湖最优 arm 补扫 μ_P ∈ {0.01, 0.03};若出现 R7-式发散(漂移单调涨且 fused 崩)→ 不加扫,记录并依赖 early stopping,该 lr 档判负。

**验收与决策**:
- 每湖:在满足 epoch-0 gate 的 epoch 里取 fused R@10 最大者;与 r5 锚点(Enti 37.74 / WDC 64.36)做配对 bootstrap。**胜出 = 点值 ≥ 锚点且 CI 下界 ≥ −0.02** → 进 Task E。
- 判读规则(供 Task D 触发):(a) P 动了且赢 → 线性解冻足够;(b) P 动了但两湖都不赢 → **函数类不足的证据,触发 Task D**;(c) P 没动 → 先走 A3,A3 后仍是 (b) 再触发 D;(d) WDC 回退、Enti 赢 → 按湖采纳,WDC 保持 r5。

---

## Task B:student-edge KD warm-up 回归(瞄准 evidence 通道)

**理由**:见背景 3/5。恢复形态**不是**把旧 in-batch 对比 edge loss 加回来(那正是伤 7pt 的东西),而是按方案.md Stage C1 的原设计:**edge-level KD warm-up → path-level fine-tune**。edge 阶段的 KD 目标与 path 阶段同形(`τ·z(s_T^edge)+(1−τ)·z(cos)`,teacher edge logits 由 `teacher_logits.py` 输出),不开 in-batch(R1 数据点:edge KD=0 无 in-batch ≈ 中性 +0.18pt;in-batch 才是 −7pt 的来源)。

**可实施性(已核实)**:τ 混合目标对 edge 列表已被现有代码支持——`teacher_logits.py:280-289` 按 example 类型分 `edge`/`target` 两种 ensemble 缓存变体(`-ensemble-edge-v*-a<alpha>`),`train_student_edges` 经 `_edge_teacher_scores` 消费之;本任务**无需新的 KD 代码**,只需在 edge 阶段传与 path 相同的 `--kd-target-teacher-alpha`/`--teacher-checkpoint`/`--teacher-logit-cache`。

**输入**(全部已存在,无需重建):
- 每湖 `edge_lists.jsonl`:Enti `work/stage1_stage2_entitables20k_v4_20260827/stage1_data/edge_lists.jsonl`;WDC `work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data/edge_lists.jsonl`。
- Teacher:模板脚本 case 分支中的 Task-M teacher(注意 edge KD 需要 teacher edge 阶段输出;Task-M teacher 带 `teacher_edge`,unification taskB 亦有重训版,取与 r5 蒸馏链一致的那枚并在 RESULTS 里记录 SHA)。

**实现**:
1. `train_stage1.py student-edge`:PCA-1024 初始化、**不冻 P 与冻 P 两种都不对——本任务先冻 P**(单变量原则:B 只测 edge 阶段本身;P×edge 的交互留给 Task E),`--freeze-projection`、`--relation-learning-rate 1e-5`、KD 0.3、τ 同湖、μ=0.1、**不开 in-batch**,edge epochs ∈ {2, 4}(剂量控制,防 R1 式几何破坏累积)。
2. **edge 后、path 前的强制检查点**(R1 教训——破坏要在进入 path 前抓住):对 edge-warmed checkpoint 直接跑一次 dev 检索评测;**若 fused 相对 PCA epoch-0 的配对 CI 下界 < −0.02,该 arm 就地终止**,不进 path。
3. 通过检查点的 arm:从 edge-warmed checkpoint 起跑 `student-path`,其余逐 flag 同 r5 模板。
4. 终评额外跑一次 r8-R2 式 4 组合零训练重组(reference-only),检查双赢门禁。

**规模**:2 湖 × 2 edge 剂量 = 4 条完整链(edge + path);每条链两次评测。

**验收**:
- primary:fused ≥ r5 锚点(CI 口径,同 Task A)。
- **secondary(本任务的真正目标)**:Enti evidence R@10 ≥ **10%** 且 coverage@10 > 3.41%,fused 不劣于 r5——若达成,r8 差 0.19pt 的双赢门禁被翻过,Enti 主结果可升级为 37.95%+ 的 fused+evidence 双赢配置。
- WDC:evidence/coverage 变化作 reference 记录,gate 只看 fused。

---

## Task C:hard-negative mining round 1(按 R2 遗留条款解冻)

**理由**:见背景 4。这是检索模型的标准自举步骤,基础设施完整在位且带指纹校验(`refresh_stage1_hard_negatives.py` 两步式、`mining.py` 三池合并、`train_stage1.py:408-440` sidecar 校验:Teacher 指纹 / 挖掘 checkpoint / 路径聚合三项必须匹配)。

**实现**:
1. **挖掘(两步)**,每湖以 r5 selection 的 checkpoint 为挖掘源(`work/stage1_optimization_r5_20260829/task4_final/selection.json`:Enti SHA `06d098c1…` = `task1_kd_tau_ablation/entitables/tau_1.0/student_path.pt`;WDC SHA `8bb317f5…`):
   - 步 1 `--pending`:从该 checkpoint 的 student ANN 索引挖 `--hard-targets-per-query 16 --hard-evidence-per-type 16 --hard-paths-per-query 16 --direct-k 200 --evidence-k 100 --mining-round 1`,**聚合 flags 与训练侧一致(`--evidence-aggregation logsumexp --evidence-top-k 4`,sidecar 校验会拒绝不匹配)**;
   - 步 2 Teacher 打分:本湖 Task-M teacher,产出 `--output-target-lists` + `--output-edge-lists` 与 metadata sidecar。
2. **重训**:从头(PCA 初始化)跑 r5 模板 + `--hard-data <mined_target_lists> --hard-source-checkpoint <对应 r5 checkpoint>`,`--hard-fraction` ∈ {0.25, 0.5}(`--hard-learning-rate` 用默认 2e-5),其余不动(`--freeze-projection` 保留——单变量原则同 Task B)。
3. **顺序与 WDC 触发条件**:先跑 **C1 = EntiTables**(teacher gate pass,打分可信)。**C2 = WDC 仅在 C1 非灾难时启动**(判据:C1 两个 hard-fraction 中至少一个 best fused ≥ 37.42,即不低于纯监督锚点),且 WDC 只跑 hard-fraction 0.25(teacher 是已知噪声源,小剂量)。
4. RESULTS 记录挖掘池构成(三池各自条数、与既有 base 列表的去重率),供论文 mining 段引用。

**规模**:挖掘 2 湖 × 1 轮;重训 Enti 2 + WDC ≤ 1 = ≤ 3 条链。

**验收**:同 Task A 的按湖 CI gate;胜出者进 Task E。若两湖都不胜,记录"mining 在健康基座上仍无增益"为诚实负结果(与 R1 的"崩坏基座上无增益"区分开——这是两个不同的结论)。

---

## Task D(条件触发):P 的非线性容量——零初始化残差 MLP

**触发条件(显式,缺一不跑)**:Task A(含 A3,若触发)完成;且 EntiTables 上 A1/A2/A3 无一胜出;且漂移遥测显示 P 确实动了(终值 ≥ 0.01)——即"优化发生了但线性函数类不够"。P 压根没动的情形回 A3,不进 D。

**理由**:R7 已证明 R 侧结构容量不是瓶颈,容量若有意义只能在 P 侧;零训练探针显示单纯加宽(2048)上界很小,**真正的新函数类是非线性**。设计直接复用 R7 低秩残差的成功要素:零初始化残差分支保证 step-0 严格 = r5 epoch-0,天然兼容 epoch-0 gate;**不做**随机初始化整表替换(那是 initialization A/B 的老路)。

**实现**:
1. 结构:`P_t(x) = P_pca x + W2_t · GELU(W1_t x)`,每 object type 独立分支;`W1: 4096×h` Kaiming 初始化,`W2: h×1024` **零初始化**;新 flag `--projection-residual-hidden <h>`(代码+测试:step-0 输出与冻结 PCA 完全一致、分支参数进投影参数组)。
2. 正则:分支输出的 batch 均方 `μ_D·E‖W2·GELU(W1 x)‖²` 记入 anchor 项(μ_D 沿用 0.1 起步);`projection_drift` 插桩改记分支输出 RMS。
3. runs:**只跑 EntiTables**(触发即说明 Enti 没赢;WDC 不追加自由度),h ∈ {256, 512} × 分支 lr 1e-5,R 冻结与联训两臂中**只取 Task A 里表现较好的那种设置**,= 2 runs;其余同 r5 模板。

**验收**:同 Task A gate。胜出进 Task E;不胜则记录"P 容量不是瓶颈"关闭该线——届时 R1–R9 将构成"encoder 冻结下,student 侧全部三类自由度(P/R/数据)均已系统排除或采纳"的完整归因链。

---

## Task E:联合验证与定稿(R6 教训的强制执行)

**理由**:R6 的核心教训——fusion/聚合/权重各自单独为正、合成后 WDC −1.49。A/B/C(/D) 的胜者**必须联合重训并重新过 gate**才能替换 r5;此外 A(解冻 P)与 B(edge warm-up)、C(mined lists)存在真实交互(P 解冻改变几何;mined 负样本是在冻结 P 的 epoch-0 几何下挖的——好在所有 arm 的 epoch-0 几何相同 = PCA identity,挖掘源与 `--hard-source-checkpoint` 声明一致即可,sidecar 校验通过)。

**实现**:
1. 每湖取 A/B/C(/D) 的胜出干预做**组合重训**(一次完整链:若 B 胜则先 edge warm-up,再 path;path 带胜出的 P 设置与 hard-data)。若某湖无任何单项胜出,该湖直接保持 r5,不做组合。
2. 组合结果与"最优单项"、r5 三方配对对比:**若组合 < 最优单项(贪心教训重演),回退采纳最优单项**,并把交互效应记录进 RESULTS。
3. 终评:primary 用 r5 检索配置(训练一致);附 r8 式重组 reference 行;两湖各产出 raw / student / +重排 三行主表 + 索引体积 + 单配置延迟。
4. **替换规则(与历轮一致)**:按湖独立采纳——某湖胜出配置 fused ≥ r5 锚点且配对 CI 下界 ≥ −0.02 才替换该湖 primary;允许一湖替换、另一湖保持 r5。FINAL.md 若无湖达标,**有意不声明通过产物**(沿用 r7 惯例),r5 继续为默认。

**验收**:FINAL.md 主表 + 每湖明确的采纳/保持决定 + 负结果段落更新(在 r8 保留清单上追加本轮新产生的)。

---

## 执行顺序与决策树

```
Task A0(插桩,半天)
  ├─→ Task A(解冻 P,8 runs;A3 条件子扫)──┐
  ├─→ Task B(edge KD warm-up,4 链)────────┤  A/B/C 互相独立,可并行占卡
  └─→ Task C(mining,C1 先行,C2 条件触发)─┘
Task A 判读 (b)/(c) → Task D(仅 Enti,2 runs)
全部单项完成 → Task E(联合重训 ≤2 链 + 定稿)
  ├─ 组合胜 → 替换对应湖 primary
  ├─ 组合 < 最优单项 → 采纳单项,记录交互
  └─ 无单项胜 → r5 保持默认,FINAL 不声明通过
```

预算上限:训练链 ≤ 20 条(A 8 + A3 ≤4 + B 4 + C ≤3 + D ≤2 + E ≤2),全部 12-epoch 缓存特征训练,量级与 r2/r4 相当。

## 产出要求

- 目录 `work/stage1_optimization_r9_20260901/task{A0,A,B,C,D,E}_*/`;RESULTS.md 逐任务追加,轮末 FINAL.md。
- Task A 交付投影/关系漂移轨迹 CSV(论文机制图备用:P 漂移 vs fused 的关系是"解冻 P"一节的核心证据);Task B 交付 edge 后检查点评测表;Task C 交付挖掘池构成统计;所有关键对比附配对 bootstrap CI。
- 新增代码(A0 插桩、A3 μ_P flag、D 残差分支)全部配测试;向后兼容(旧 checkpoint 加载行为不变)。
- 诚实负结果照旧保留:任何 arm 的失败按"哪个假设被证伪"记录,不合并进成功叙事。

---

## 数据集重建的代码适配(前置任务 F,先于一切训练 arm)

### 背景:数据集语义变化(已在 joinability.py:134 / build_mm_joinability_dataset.py:7005 / extraction.py:181 / wdc200k_materialize.py:7191 落地)

用户对数据集做了以下修改,**改变了 Stage-1/Stage-2 的训练与评测假设**:

1. **一个 query 可有多个正例**(链接列与 target context 列独立打乱,join 列不再固定第一列;所有合格 join 列都可生成 target;可见列/行/源行集相同的 query 合并,`hidden_attributes`/`target_table_ids`/`chain_ids` 保存全部路径,生成多条 qrel;仅拒绝重复 qrel,允许一个 query 多个不同正例)。
2. **普通列按源表确定性打乱**,target 比例从 `N(0.5, 0.1)` 采样裁剪 `[0.3,0.7]`,query/target 两侧尽量都有普通列;`max_query_context_attrs`/`max_target_context_attrs` 数量限制失效(旧参数保留为兼容 no-op)。
3. **bridge 列排除出普通 context 池**(避免泄漏或未标注正例)。
4. `extraction.py` auto-check 改为按 `(query_id, target_id)` 独立统计(一个属性的恢复结果不再错误保留另一个 target)。
5. **新版 WDC 阶段指纹加入构造策略版本**(防止复用旧单 target 物化缓存)。

### 核实:现状与缺口(× = 需改,✓ = 已兼容)

| 模块 | 现状 | 判断 |
| --- | --- | --- |
| `data.py` 加载 | `TargetExample.positive_target_ids`(tuple)已存在,加载时校验 designated ⊂ positive_target_ids | ✓ 数据层已支持 |
| `evaluation.py` 评测 | `positive_sets = set(example.positive_target_ids)`,recall/mrr 按集合算,evidence hit 按 candidate.evidence_ids 匹配 | ✓ 多正例正确 |
| `scoring.py` in-batch | `:539` 用 `set(example.positive_target_ids)` 排除正例 | ✓ 已支持 |
| `mining.py` 挖矿排除 | `_known_positive_target_ids` 用 `positive_target_ids` 排除 | ✓ 已支持 |
| `construction.py` 生成 | 已写 `positive_target_ids`(全部正例)+ designated 单值 | ✓ 生成侧已支持 |
| `oracle.py` / `stage2 data.py` | 显式用 qrel 的 `source_column_index` 做 `local_column_index` 映射,不假设第一列;`qrel_pairs` 按 `(query_id,target_id)` 去重 | ✓ 已兼容多 target |
| **训练主 loss**(`objectives.py` + `training.py`) | **`listwise_cross_entropy`(单正例 softmax)、`_path_supervised_losses`、`_target_teacher_scores` 全部只用 `direct_positive_index`/`evidence_positive_index` 两个单正例** | **× 核心缺口** |
| `teacher_rerank.py` | `:337` 用 `positive_target_ids` 集合 | ✓ 已支持 |
| `retrieval_aligned.py` | `:134-226` 写 `direct/evidence_positive_target_id` + `positive_target_ids` | ✓ 已支持 |
| `features.py` / `cache_stage1_features.py` | 表输入 = `table_parts`,不假设 join 列位置;`source_fingerprint` 按记录内容哈希 | ✓ 数据集重建自动触发特征重缓存 |

**关键结论**:数据生成、评测、挖矿、Stage-2 oracle 都已走"多正例集合";**唯独训练主 loss 仍走"单正例 index"**。这意味着在多正例 query 上,训练 loss 只看"任意一个正例"而评测看"全部正例"——训练目标与评测口径不一致。

### Task F1(必须,训练主 loss 支持多正例)

**目标**:让 `listwise_cross_entropy` 及其训练侧调用链均匀支持多正例集合。

**实现**:
1. `objectives.py`:
   - 新增 `positive_mask: torch.Tensor`(shape `[batch, n_candidates]`)作为可选参数;`listwise_cross_entropy` 把它解析成 multi-positive CE(对所有正候选的 `softmax` 概率之和取 `-log`,即 `F.cross_entropy` 的 `target` 为 soft 多标签,或用 `-log(sum(softmax[positive]))`)。
   - 保留单正例 `positive_indices` 分支(向后兼容),并新增一个 `positive_indices` → `positive_mask` 的转换辅助函数(one-hot 展开)。
   - `_usable_list_rows` / `optional_listwise_cross_entropy` 改为以"至少一个正候选且至少一个负候选"为行筛选条件。
2. `data.py`:`TargetExample` 已在加载时把 `positive_target_ids` 解析成 set(缺失时 fallback 到 designated),用它在训练批组装时生成 `positive_mask`。
3. `training.py`:
   - `_path_supervised_losses` 与 `_target_teacher_scores` 改为传入 `positive_mask`(由每个 example 的 `positive_target_ids` 展开)。
   - `student_evidence_anchor_loss` / `_path_distillation_losses` 同期适配(若它们也走 `positive_indices`)。
4. `scoring.py`:`score_target_batch` 与 `_score_student_candidate_rows` 的 positive 从单 index 改为 `positive_mask`(in-batch 时 mask 按 `positive_target_ids` 展开)。
5. **测试**:
   - 单正例路径输出与旧实现一致(回归保护)。
   - 手工构造一个多正例 query:训练 loss 应把两个正例都计入(对比单正例 loss 的差异)。
   - `_usable_list_rows` 在有正例但全为正(无负候选)时跳过该行。

**验收**:单正例回归测试通过;新增多正例测试通过;加一次 1-epoch smoke run 确认训练不崩、loss 有限。

### Task F2(必须,dataset 其他 side-effect 校验)

**目标**:处理数据集重建的连锁副作用,保证实验可比。

1. **`max_query_context_attrs`/`max_target_context_attrs` no-op**:`joinability.py` 中这两个参数不再限制快照列数。旧参数保留为 no-op,命令不报错(已在构建脚本中完成,这里只需在 Stage-1 侧确认**无代码硬依赖"可见列 ≤ 某上限"**)——已核实 `cache_stage1_features.py` 按 `table_parts` 全量输入,无数量上限假设,✓。
2. **bridge 列排除**:`_balanced_context_partition` 已排除 bridge 列。**需要确认**训练用的 `table_parts`/特征缓存不含被排除的 bridge 列——因为特征缓存按记录内容 fingerprint,数据集重建后 fingerprint 不匹配会自然重缓存,✓;但**旧缓存目录必须删除或换新目录**,避免用旧 fingerprint 的旧 embedding。
3. **WDC 阶段指纹含构造策略版本**:`wdc200k_materialize.py` 已加入构造策略版本。Stage-1 侧需确认读取 WDC 数据的指纹/版本校验(若有)接受新版本——**若读取侧有硬编码版本号,需同步更新**。
4. **`extraction.py:181` auto-check 按 `(query_id,target_id)` 独立统计**:确认 Stage-1/Stage-2 读取 `evidence_recoveries`/qrel 时,`recovery_evidence` 是按 query 聚合的(如 `construction.py:362` 的 `recovery_evidence` 推导)——**若 recoveries 现在按 target 去重,`construction.py` 的 `evidence_positive_target_ids` 推导逻辑需重审**,确保不因"每 query 多 target"而把某个未标注 target 误判为正例。

### Task F3(必须,重建数据 + 校验多正例数据流)

**目标**:用新版数据集重建 Stage-1/Stage-2 数据,验证多正例在真实管线中端到端生效。

**实现**:
1. 重建 EntiTables + WDC 两湖的 `stage1_data`(target_lists/edge_lists/corpus/objects)与特征缓存(新目录,旧缓存不动)。
2. 校验:
   - **target_lists 多正例统计**:每 query `positive_target_ids` 长度分布(与旧数据 全为 1 对比)。
   - **train/dev 一致性**:同一 query 的 `positive_target_ids` 在 train 与 dev 中一致(合并 query 可能有跨 split 的正例——若 query 跨 split,需明确 split 归属规则)。
   - **candidates 覆盖全部正例**:每个 query 的 `candidates` 必须包含全部 `positive_target_ids`(否则 `data.py` 加载会报"designated 缺省"错,且评测 recall 分母被截断)。
   - **重复 qrel 校验**:同一 `(query_id,target_id)` 不重复;同一 query 多 target 时 `direct_positive_target_id`/`evidence_positive_target_id` 仍在 `positive_target_ids` 内。
3. **Gate 锚点重排**:因为数据集语义变了,旧 raw 基线(Enti 30.92/WDC 62.38)不再适用。**必须先在新数据上重跑一个零训练的 raw direct R@10 / R@50 作为新的 epoch-0/raw 参照**,否则后续所有 gate 都在旧口径上比。
4. 若 multi-positive 导致训练集/评测集规模变化,记录新的 query 数,并重算每湖 dev 正例基数(1pt = 几条查询)。

**验收**:多正例统计符合预期;每湖新 raw 基线落盘;旧基线标注为"历史口径,不可比"。

### Task F4(条件):Stage-2 若受影响,同样适配

**触发**:若重建后 Stage-2 训练/评测出现多 target 相关错误或行为变化(如 oracle 正例数变化、reader 需处理多 join 列)。

**实现**:审查 `stage2/qwen.py` `training.py` 消费 qrel/positive 的假设;确保 oracle / reader 对"一个 query 多个 target(不同 join 列)"的处理正确。

---

### 对 Task A/B/C 的影响(前置依赖)

- **所有训练 arm(A/B/C/D/E)必须在 Task F1(F3 的 loss 适配)+ F3(新数据重建 + raw 基线重锚)完成后才能开跑**,否则训练在多正例数据上用单正例 loss、且 gate 锚点还是旧 raw,结果不可比。
- **Task A/B/C 输入路径全部指向新重建的 stage1_data/特征缓存目录**,不再用 `r4/taskJ_per_lake_baselines/...` 旧路径。
- **旧 r5 基座依然有效作为"旧口径"的锚**,但新数据的 r5 重训是新的参照——**建议 Task A 之前先加一个 `r5_repro_on_new_data` 臂**(在无任何干预下,用新数据重跑一遍 r5 模板),确认新数据下 r5 配方的表现(可能与 37.74 不同),再以此为新的比较基线,否则无法判别"是 P 解冻的增益还是数据的增益"。
- **mining 的 sidecar 指纹校验**:Task C 挖掘的 `--hard-source-checkpoint` 必须是最新重建数据上训出的 checkpoint(旧 r5 checkpoint 与新数据 fingerprint 不符,`train_stage1.py:430` 会拒绝)。

---

## 前置依赖关系(插入到执行顺序)

```
Task F1(loss 多正例) + Task F2(side-effect)─┐
                                            ├─→ Task F3(重建数据 + 新 raw 基线)
Task F0? r5_repro_on_new_data(新数据复现 r5)─┘
     │
     ├─→ Task A / B / C(新数据上跑)
     └─→ (全部基于新数据集,gate 锚点 = 新 raw / 新 r5 repro)
```
