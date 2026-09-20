# MMDD Stage1 FRESH-PATH：从原始数据与 GT 独立完成 Teacher→Student→ANN→Path 重排

**版本 2.1｜2026-09-20｜独立完整规范｜完整 fresh 主线 + 独立 QT 对照 + 双 RTX 4090 并行｜未执行的实验方案。**

> **执行者必须先完整阅读本文件及协议配置，再实现和执行。下列约束没有“为提速”“为兼容”“先复现历史高分”的隐含例外。不得自行换起点、数据范围、标签、损失、候选、训练次序、终点或最终评分。缺少输入应按本文规定重建；不能用历史训练产物补缺。确实无法执行时，报告具体阻断项，不降级，不把未完成写成完成。**
>
> **本轮必须是完整 fresh Stage1。不是在 B13 上训练一个 adapter，不是在 T0 上继续训练 Teacher，不是先做历史 parent 局部验证再决定是否从头训练。**

## 0. 文档效力与本轮唯一目标

### 0.1 替代而非叠加

本文件完整替代 FRESH-PATH v2.0、QCPATH-R1 v1.0、S0 修订 v1.1 及 CLEAN-R1 的执行指令。不要同时执行旧 A/B 状态机，不再要求历史 train_fit、B13/T0 哈希、历史 target index、旧 retained-path 文件存在。旧文档仅可作为阅读背景，不能补充本文件没有授权的默认行为。

阅读顺序：本文件 → `protocol.json` → `CONTROL_MATRIX.md` → `EXECUTION_DAG.json` / `PARALLEL_EXECUTION.md` → `ACCEPTANCE_CHECKLIST.md` → `CODEX_PROMPT.md` → `reference/`。本文解释数学与语义，JSON 锁定数值。两者不一致时停止并列出具体字段，不能挑有利的一版。用户后续明确指示可以修改协议，但必须形成版本记录；执行者自己不能修改。

### 0.2 必须交付的整链与并列对照

```text
原始带 GT 的 20K 数据集 + 原始 train/dev/test + 公共冻结 Qwen
    ↓ 重新枚举对象、原 train 监督、本轮 raw 组合、本轮 PCA
新初始化 Teacher → T_EDGE（2 epoch，五关系）
    ↓ 一次本轮难例刷新；保存不可变 T_EDGE 快照
    ├── T_PATH（2 epoch，真实 QET/Path）→ 冻结 T_PATH
    └── T_QT（2 epoch，独立 QT 排序训练）→ 冻结 T_QT

完整多模态主线（不删除原 v2.0 的训练阶段）：
  新 Student SUP/KD 各 C1（1 epoch；KD 仍由 T_PATH 提供）
    ├── SUP_C1 → SUP-QE C2（3 epoch，冻结 P/R，条件 adapter）
    ├── KD_C1  → KD-QE C2（3 epoch，预定主方法）
    ├── KD_C1  → KD-EONLY C2（3 epoch，Q 输入置零，仍读取 E）
    └── KD_C1  → KD-NATIVE C2（1 epoch，旧式无条件 ET，训练 P/R，无 adapter）

无 evidence 输入的独立 Student 对照：
  新 P_table/R_QT → QT-SUP C1（1 epoch）→ QT-SUP C2（3 epoch）
  同一新初值     → QT-KD  C1（1 epoch）→ QT-KD  C2（3 epoch；教师 T_QT）

最终：各模型 own ANN → 对照矩阵（QT-only / 真正 Path）→ 完整 dev
      → 固定 repeat gate → 至多第二 seed → 配方锁定 → 原 test 一次
```

T_PATH 与 T_QT 是**同一架构的两个独立训练分支**，不是线上串联两个 Teacher，不增加同一个模型内部的输出头。它们在本轮共同 fresh 的 T_EDGE 终点分叉，而不是 T_PATH 训练完再临时把 E 置空。正式部署主方法仍只有 KD-QE Student + T_PATH Teacher。

**三种容易混淆的对照必须同时列名：QT-only Student 不读取 E；KD-NATIVE 仍有 E→T，但完全不读取 Q 条件；KD-EONLY 仍读取 E 且有新增 MLP，只屏蔽 Q。后两者不能叫“无 evidence”。**

首个 seed 必须走完主线及全部指定对照，不能只在 Path 表现好时才训练 QT 对照，也不能只交付 QT 对照而省略 Path。除数据/实现错误、非有限数值或真实资源超限外，不因为某个中间指标没有超过历史 B13 就取消后续阶段。性能取舍在首个完整 seed 后进行。固定预算耗尽即结束，不续命式追加训练。

### 0.3 Fresh 的可操作定义

删除/屏蔽所有历史任务 checkpoint、旧负例列表、旧训练图、旧 Teacher logits、旧 PCA 文件、旧校准映射后，主命令仍能仅从原始数据、公开底座、代码与配置完成训练。允许复用有来源证明、逐对象兼容的纯冻结编码缓存，冷启动缺缓存时必须具备生成路径。

流程内部产生的 `T_EDGE→{T_PATH,T_QT}`、各自 `S_C1→S_C2` 是同一次运行中预先规定的阶段继承，不是循环依赖。每个中间权重都必须能够沿有向无环依赖图回溯至本轮初始化。

### 0.4 不做的事情

不训练 Stage2，不做属性生成或图像裁剪，不训练 200K，不新增标注项目，不调用付费外部 API，不做网格，不自动做 Student→Teacher→Student 反复反馈。Stage2 后续仍使用全局最大 `softmax(table_score)×column_score`；本轮只交付其需要的表分和证据，不改该规则。

### 0.5 核心约束表

| 编号 | 必须满足 |
|---|---|
| F01 | 原始 dataset train 是全部允许的训练监督范围；不继承历史 train_fit/cal_fit/cal_check。 |
| F02 | Teacher 的全部任务参数从新初始化开始，包括 pooler、adapter、global 分支、角色、Transformer、head。 |
| F03 | Student P/R 从本轮重算 PCA/identity 初始化；条件模块新初始化。没有外部任务权重。 |
| F04 | 所有新增训练组合、难例、图、分数在本轮生成，禁止读旧列表后改名。 |
| F05 | 公开 Qwen 冻结；只离线生成缺失的纯内容特征，在线不调用 Qwen。 |
| F06 | Query 所有 example rows 独立保留；不截前七/十二行，不跨行平均。 |
| F07 | 一种 Teacher 架构、每个模型一个共享输出头；T_PATH/T_QT 是独立对照权重，不联训、不混分，不恢复 P/J 双任务或 gate。 |
| F08 | 最终按同一 Teacher 的零跳与真实 QET 路径聚合排序；不接外部 QT 分数，不回退 QT-only。 |
| F09 | Student 每目标一个静态向量；第二跳只改变查询侧 v(Q,E)，不得按 Q 构造目标索引。 |
| F10 | 条件化主线C2使用完整合法目标竞争与ρ=0.5；QT-only C2也用全目标SUP但无adapter；KD-NATIVE按专章，不误套冻结/rho。 |
| F11 | 保留完整 Teacher→Student 蒸馏，包含边、条件化第二跳和目标路径分布；同时有 SUP 对照。 |
| F12 | 原始 GT 只用于监督和评测，不输入 GT join column、恢复值、来源 ID、候选来源/排名等答案信息。 |
| F13 | 未标注项是 assumed competitor，不宣称确认负例；所有 train-known positives 受正确作用域保护。 |
| F14 | Dev/test 的标签不能进入训练正例闭包、ignore、采样或负例过滤。 |
| F15 | Raw Qwen 基线必须同协议运行；历史 B13/最强模型仅可隔离事后评测，不成为主链依赖。 |
| F16 | 不通过训练 list shape、padding、missing field 或缓存回退改变空正集语义。 |
| F17 | 不复制失败 query、删困难样本、仅保留有特征/有自然 witness 的子集来改善指标。 |
| F18 | 不以“参考测试通过”冒充仓库集成、真实训练或最终评测完成。 |
| F19 | T_QT 必须独立完成规定训练；T_PATH 的 f0-only 不能冒充它。 |
| F20 | QT-SUP/QT-KD 从本轮新 table P/R 初值训练；不得从多模态 Student 删 E 得到并声称独立训练。 |
| F21 | KD-NATIVE 使用原双线性无条件 ET，无新增 adapter；不得拿 KD-EONLY 改名替代。 |
| F22 | 全部实验硬件为 2×RTX 4090；同卡多进程、双卡独立分支并行按 §4.6 执行。 |
| F23 | 并行只改变作业调度；不改变任何作业内样本、顺序、精度、logical batch、更新量、seed 或模型/缓存归属。 |

---

## 1. 已知证据与本轮设计的边界

用户原要求是“完全抛开之前得到的任何 checkpoint 和 train data list”“用原始带 GT 数据集从头做完整 Stage1”，见随附 `SOURCES_AND_DECISIONS.md` 所列对话来源。旧 QCPATH 文档却明确限制为局部续训，本版纠正的是这个目标偏移，不把换一个 seed/optimizer 冒称 fresh。

Bridge 支持“特定旧 C1 的 356→659 续训损害 evidence 链”，不支持“任何 fresh 流程必须固定356步”。本版将 Student C1 定为完整原 train 的一次 query-balanced 遍历，并在 C2 冻结 P/R；不裁数据来追求旧步数，也不声称已验证此新预算会恢复旧分数。

旧 QC-ET 已做零初始化 adapter、冻结 P/R，但无约束残差和固定小分母仍出现公共目标。完整分母与受限残差在本版保留。它们同时变动，不能把相对旧失败实验的收益归因于某一个因素。

本版新增对照用于回答“fresh 旧式结构能否接近历史工作点”和“E 是否产生额外收益”。这不是把 QT-only 换回主方法。历史 checkpoint 不用于训练、不用于补齐监督，也不作为阶段门槛。

本版不是历史 B13 的精确复现。Teacher 从新初始化、训练范围为完整 train、候选与表示合同有明确新定义。所有超参数是一次预先确定的工作点，不是已有数据证实的最佳值。最终可用性必须由真实整链实验决定。

## 2. 输入与来源：不能再锁死到历史目录

### 2.1 仅允许的根输入

1. 原 20K 数据集快照：query、target/lake tables、text/image assets、原始 split、qrels、query-specific evidence/recovery GT，以及展开引用表所需原始对象。
2. 公共 Qwen3-VL-Embedding-8B 权重、tokenizer、processor 与官方 embedding wrapper。使用本机现有版本，记录实际 revision/文件指纹；没有本地权重时列出缺失，不擅自联网下载另一模型。
3. 本版代码与配置。
4. 可选纯冻结编码缓存：只加速，不决定对象/训练人口；不包含任何任务训练参数、Teacher 学习压缩结果或旧筛选名单。

数据集从 `dataset_manifest.json` 或原始 JSONL schema 解析；路径可沿此前 S0 已确认的原始dataset/backbone位置定位，但不能继承该S0的旧parent或train_fit角色。对象集合和 split 必须来自数据集，不来自某个 `work/rXX/...` 训练 manifest。相同内容被存放在 work 目录并不自动禁止；必须依据生成来源，而不是文件名判断纯缓存是否允许。

### 2.2 输出 `ROOT_INPUTS.json`

每个角色记录：真实绝对路径、用途、字节数/目录清单指纹、来源类型、是否允许训练读取。必需角色为 dataset、raw_splits、raw_train_GT、raw_dev_GT、raw_test_GT_location、public_backbone、encoder_contract、code、protocol；pure_cache 可以为空。

**不得新增必需角色 `student_base`、`teacher_parent=T0`、`historical_PCA`、`old_train_fit`、`old_hard32`。**读取这些角色作为训练输入应直接失败。

原始 test 的湖中内容对象可以被无标签编码；test query可在最终评测前按同一规则编码，但不参与PCA或训练。test 相关性标签的定位可登记，正式训练/开发入口不打开其内容。

### 2.3 原始 split 的唯一规则

使用 dataset 原 train/dev/test。只在原 train 建 G/D/W/Epos 等监督。源组是否跨 split、对象是否重复做一次检查并报告；若原数据本来有共享湖对象，不能误判为 query 标签泄漏。发现确切 query/source-group 标签划分冲突，不得自行重划或移动样本。

本轮没有单独校准任务，不划 calibration，不套旧 cal_check。新论文如需独立校准是后续新协议，不在本轮补做。

“完整 train”是允许监督范围。某个 query 没有 witness 时不产生 QET 正例，但它仍用于 QT/目标排名；不得因为它没有 witness 就从整个训练集中消失。

### 2.4 旧产物隔离与恢复

主 CLI 不提供 `--init-from-b13`、`--teacher-t0`、`--reuse-old-list` 等参数。模型加载白名单仅为 public backbone 与当前 run 已完成阶段；同一 run 的 resume 需完全匹配 root/code/config/data-order/phase/seed。

允许阅读旧纯函数代码以复用算法实现，但不能调用其带历史 `_paths()` 的 runner。搬用旧模块前列出其读取的文件；导入时读取旧权重/旧列表也算违规。

所有本轮中间产物带 `run_id, protocol_hash, root_hash, stage, model_seed, parent_artifact_ids`。跨 run 的任务参数即使同名、相同尺寸或相同 seed 也不能加载。参考测试不能证明不存在隐式文件读取；仓库必须在受限 loader/根路径隔离下通过真实 smoke。

---

## 3. 监督重建：G/D/W 与三种关系作用域

### 3.1 原始标注映射

- `G[q]`：原 train 中 q 的全部正确 target。
- `D[q]⊆G[q]`：原 GT 明确合法的 direct target；若没有独立字段，只能按原生成器显式记录的 explicit/direct 事实映射，不能由模型 QT 分数推断。
- `W[q,t]`：原始 query-specific witness/recovery 记录中支持当前 q,t 的 evidence 集合。保留自动审核/人工审核的真实等级和原始记录位置。
- `Qpos[q,modality]`：对应模态上所有 `W[q,t]` 的并集。
- `Epos[e]`：原 train 全部 W 中 e 对应目标的并集，加上原数据确有、明确属于 train 的无条件 ET 正标注。
- `P[q,e]={t:e∈W[q,t]}`：当前条件路径的正目标。

绝不从旧训练列表推断 D/W。相同 target 在其他 Q 的 witness、同源网页或相同 URL 不构成当前 q,t 的 W。缺 recovery 保留缺失；不能以另一个 Q 的 recovery 填充。

同一(q,e,t)重复 GT 合并，不因重复标注增加 loss 权重。空数组、字段缺失、null 有不同含义；导出时保持显式字段，不以 `if payload.get('positive_ids')` 决定是否恢复。

### 3.2 数据对象与模型输入分离

GT 字段可用于构造标签、训练增强和指标，不能拼进 embedding prompt、Teacher token、Student vector。表本来可见的正常列名/单元格不是泄漏，照常保留。target ID、source ID、GT join-column index、gold recovered value、raw rank、candidate source 只存在审计元数据中。

Qwen 不能读取 mask 前被隐藏的属性或源表中未提供给 query 的额外行。引用表展开只用于恢复原本属于该对象的内容，不用于“补回 query 缺失属性”。

### 3.3 合法湖对象与闭包

合法目标库来自原数据集的 candidate universe 和明确的非标签自表排除规则；至少排除 target_id 等于 query_id 的真正同一对象。不能因为对象 ID 前缀为 `dl_raw_`、属于同来源或未标正就删除。

若数据集未定义额外排除规则，不自行发明“同 source 一律排除”。在 §4 manifest 锁定合法性函数。GT 正目标被该规则排出是数据/实现冲突，应阻断，不能删 GT。

QET 的完整合法库 mask：
\[
P_{q,e}=P[q,e],\quad I_{q,e}=((G[q]\cup Epos[e])\cap\mathcal T_q^{legal})\setminus P_{q,e},
\]
\[
N_{q,e}=\mathcal T_q^{legal}\setminus(P_{q,e}\cup I_{q,e}).
\]
P 进入分子与分母；I 从该项 loss 的分子/分母同时移除；N 是 assumed competitors。P 必须非空且属于 legal；N 空则该条件项 inactive，两臂一致记录。

这里保守忽略了同 E 其他 Q 已知关联，**不保证标签足以强迫模型使用 Q**；E-only 对照负责检验 Q 增量。禁止为了制造 Q 有用而把 I 自动变负。

### 3.4 五个 edge 任务

| 任务 | Anchor | 正集 | 竞争库 |
|---|---|---|---|
| QT | q | G[q] | 合法 target |
| Q→text | q | Qpos[q,text] | 全合法 text |
| Q→image | q | Qpos[q,image] | 全合法 image |
| text→T | train 关联 e | Epos[e] | 合法 target |
| image→T | train 关联 e | Epos[e] | 合法 target |

无条件 ET 使用 Epos，不把当前某个 Q 的标签差异误写成无条件 ET 负例。QET 使用 P/I/N，是另一种输入条件；两者不能复用错误 mask。

每个 q 的 edge 组包括 QT、非空的两类 QE、q 的 W 涉及的每个不同 e 的 ET。ET 在同 q 去重；跨 q 可重复，但先在 q 内按关系平均，再按 q 平均，防止 witness 多的 query 支配训练。

### 3.5 本轮监督产物

输出 `labels/train_queries.jsonl.gz`、`train_conditional_pairs.jsonl.gz`、`edge_registry.jsonl.gz`、`label_stats.json`、`label_conflicts.jsonl`。记录完整原 train 数、各项活跃数、缺失原因。14592/16156、5707/6315 只是此前报告线索，不作为断言，更不能按它们筛样本。

评测 qrels 在独立进程读。改变 dev/test 标签但不变内容后，本轮 train G/D/W/Epos/P/I/N、PCA和训练 candidate ID 必须不变。

---

## 4. 纯冻结表示与资源：缺缓存能生成，不能依赖旧 Teacher

### 4.1 基础编码合同

底座为本地公共 Qwen3-VL-Embedding-8B，参数冻结、eval、inference_mode，BF16 前向；输出最后有效 token 的 embedding，float32 L2 normalize 得 z∈R4096。不得用“末尾一个固定 padding token”取代最后有效 token。

沿用源码 `cache_stage1_features.py` 的 `role_modality_v2_object_only` **五条文字提示**，副本见 `ENCODER_PROMPTS.json`；本包直接从已上传源码提取，不以旧缓存中的未知 prompt 为权威。query、target、text evidence、image evidence 的提示不同但只描述对象角色，不含 GT。

固定 max_tokens=8192。图像 resizing/min_pixels/max_pixels 与 tokenizer/template 使用本机公共模型 processor 的明确配置值，写入 `ENCODER_CONTRACT.json` 后锁定；不能运行时搜索哪个 resize 得分好。缺配置无法确定时报告具体字段，不读取历史模型成绩决定。

表序列化为 `Columns: c1 | c2 ...`，每行为 `Row: v1 | v2 ...`；值由原数据 text 字段取出，None→空字符串、统一换行/连续空白为单空格、strip，保留大小写和 Unicode，不用 ID 代替正文。

Query：全部原始 example rows、全部可见列，单元格不额外截字。Target：按原数据顺序前12行，单元格最多1024字符；这是目标内容抽样规则，不是 query 规则。分别记录 query/target 行数，禁止混合直方图掩盖 query 截断。目标所有列保留。

Query/target 完整表输入超8192 token 时，先报告超长对象并阻断对应准备，不默认再截行或用旧多次缩短重试。文本 evidence 超长允许固定保留 wrapper 的同一前缀8192 token，记录截断；不为正例额外保留 GT 附近片段。图像按 processor 单次确定处理。无正文、坏图、无法展开表不补零：准备期修复或阻断，不能静默缩小湖。

### 4.2 Teacher 输入缓存，明确区别于旧固定八槽

本版为受限磁盘规定**纯底座、内容级压缩**，不缓存任何学过的 Teacher 输出：

- Query：一个 schema 摘要 + 每个 example row 独立摘要，不跨行平均。
- Target：一个 schema 摘要 + 每个已序列化行摘要，最多12行。
- Text/image：从这次前向中只选正文 token/图像 token 的最后层状态；去掉系统提示、角色模板和 padding，保留 token 顺序。若有效 token 数 L≤64，逐 token 保留；L>64，分成64个连续非空桶，每桶 float32 均值。最后转 float16 存储。

桶 j 范围为 `[floor(jL/64), floor((j+1)L/64))`。完整有效 token 只在这次离线前向/旧纯缓存读取时暂存，处理完释放，不新建全湖全 token hidden-state 库。

**这是本版明确的资源工作点，不声称与历史 Teacher 全 token 输入等价，也不是重新使用 CLEAN-R1 的固定8槽。**表尤其是 Query 不套64桶规则。用户关注的每个 Query 行仍有独立 token。

输入宽度仍4096；Teacher 会在训练中用新的 adapter 和 learned pooler处理这些底座摘要。底座摘要固定可缓存；正在训练的 adapter/pooler 输出不可跨 step 缓存。

文本字符 offset、图像 token mask、表 row span 必须与实际 `input_ids/attention_mask` 对齐；不能通过猜固定token区间选正文。若无法对齐，报告失败，不拿全 prompt 平均替代。

### 4.3 旧纯缓存的允许条件

只有对象内容、序列化、角色prompt、公共模型版本、processor、最后层位置、mask、pooling 和量化顺序均兼容，才能复用。原全 token 纯底座缓存可按4.2压缩；旧固定8槽不能反推64槽，旧 Teacher 的learned 16/24 latent不能当原始hidden states。只有z时可复用z，缺的局部摘要另行生成并验证z一致性。

对象全集从 dataset 枚举。缓存缺132个、13200个或任何数量，都不是按历史身份直接判方法阻断的依据：计算实际需要范围和资源后，离线补齐，遵守资源预算。没有“最多2048个历史漏项”的强制旧限制。

### 4.4 按需准备，不按缓存筛样本

先生成全湖 z（缺失者一次前向同时保留其本版摘要，避免将来再跑），再产生本轮 raw 训练候选；所有Teacher训练和蒸馏所需的q/t/e都必须覆盖。后续Student own-pool评测出现新e时，在评测前同配方补齐，一次生成后复用。

如果已有z但无局部摘要，允许针对需要对象额外一次冻结前向；这叫补齐，不能声称整个流程每对象总共只前向一次。禁止每个epoch重新跑Qwen，禁止为每个(q,e,t)单独重编码对象。

最终需要所有 query 行的独立表示；原始模型输入中实际被截掉的行不能用复制行摘要补齐。原始 z 与Teacher行摘要的输入内容必须匹配。

### 4.5 磁盘、内存与硬件

本轮新特征上限128 GiB，新增run总上限160 GiB，保留至少200 GiB可用磁盘。不删除旧实验、不覆盖原cache。64×4096×2=512 KiB是每条text/image局部缓存上限，另加16 KiB的z；按真实全对象数量预算，不能只乘目标表数。

**用户确认本项目从始至终均使用 2×RTX 4090，每卡24GB。本文只使用这一硬件合同，不把历史模型归属于其他设备。**每个优化作业只绑定一张物理卡，不启用 DDP 或模型并行；独立作业可以分配到两张卡，也可以经 §4.6 验收后共用一张卡。用户观察现有训练/推理利用率偏低，作为调度优化动机记录；实际峰值显存、吞吐和利用率必须现场测量，不能假设所有阶段都低负载。

Teacher/Student任务参数与loss全float32，autocast/TF32关闭。Qwen是独立BF16冻结提取配置，不混成“所有训练是BF16”。CPU特征LRU最多8 GiB、预取进程最多4。Qwen和其他核心库不自动升级。

若完整需要超过预算，列出对象/张量/模型/排名的实际字节分解并 `BLOCKED_RESOURCE`；不偷改64→8，不裁query/target/evidence，不改到旧parent。资源不足是未执行，不是方法失败。


### 4.6 双卡与同卡多任务调度合同（必须执行，不把低利用率当成串行理由）

**调度优先级：先让两张卡各有一个已就绪的独立作业；如果仍有就绪作业且显存/吞吐条件满足，再尝试同卡两进程。默认上限每卡2个任务、全机4个任务。**不按平均显存或瞬时0%利用率直接多开。下列阈值是运行资源策略，不是训练超参数。

1. 每类任务先用独立临时模型完成2个warm-up更新和8个测量更新（推理任务用2+8个固定请求），覆盖已知较大候选/路径样本。记录独占时峰值 `torch.max_memory_reserved`、可获得的设备进程显存增量、step时间和总完成时间。临时更新不进入正式权重，单独记录其计算开销。
2. 每任务显存预约值取 `max(框架峰值reserved,可测进程显存峰值增量)+1 GiB`。所有共驻任务预约量加其他进程占用，必须同时不超过实测设备总量的85%，并留下至少3 GiB空闲。监测总设备显存而非只看PyTorch allocated。
3. 满足内存条件后，允许两独立临时进程共驻试跑相同固定工作量。对比“两个作业独占依次完成的总时间”和“共驻完成的makespan”，加速比至少1.05且无错误才保留共驻；否则恢复一卡一作业。两者都使用相同微批、样本和精度，不能通过缩小训练任务证明提速。
4. 上述试跑同时检查同一初始化、RNG、数据和配置下独占/共驻的loss、梯度及参数差异，采用 §14.5 容差。若独占自身有波动，记录实际基线；不得放宽容差使不等价实现通过。不承诺跨驱动或任意内核调度逐位相同。
5. 每作业独立OS进程、独立optimizer、模型RNG、data RNG、日志、checkpoint和可写缓存。`CUDA_VISIBLE_DEVICES=<物理ID>` 后进程内用 `cuda:0`；回执同时写物理GPU UUID/ID与逻辑device，避免两作业误落同一物理卡。禁止用设备编号或启动次序生成模型/采样seed。
6. 共享只读的纯z/摘要/PCA、已经完成且冻结的Teacher、已封存的候选文件。缓存键至少含run/teacher分支/checkpoint hash/模型分支/候选顺序/mask/view；相同 `(q,e,t)` 在 T_QT 与 T_PATH 下分数不能串用。共享文件由单一writer生成，临时文件完成校验后原子发布；读取者不读取半文件。每阶段完成回执提交后才释放依赖。
7. CPU预取worker**全机合计最多4**，LRU预算**全机合计8 GiB**；并发作业分配各自配额，避免每进程各占4 worker和8 GiB造成争抢。保持每作业既定batch顺序，不按预取完成先后训练。CPU列表预计算、去重、索引/元数据准备可与无依赖GPU工作重叠。
8. 冻结Qwen补特征默认独占其所在卡，另一张卡可执行不依赖缺项的任务。禁止为了共驻给Qwen换量化、给任务模型开AMP、缩候选/路径/序列或改变原公式。
9. 正式作业若发生OOM，停止受影响作业，保留最后完整提交的checkpoint/optimizer/RNG/数据游标；清除失败step未提交状态，按完全相同配置独占恢复。记录重算step与额外开销。不得跳batch、沿用部分梯度、换seed或删除难样本。独占同配置也不够时遵守原分块/资源阻断规则。
10. 线上时延p50/p95正式测量必须在该GPU无其他实验作业时进行；并发吞吐另表报告。不能拿共驻时延与独占历史时延当公平效率比较，不能把墙钟重叠时间相加说成总时长。

输出 `RESOURCE_PROFILES.json`、`SCHEDULE_EVENTS.jsonl`、`CONCURRENCY_PARITY.json`。不要求建设通用调度平台；薄CLI加有限进程队列即可。未通过共驻验收只关闭共驻，不取消独立双卡并行，也不阻断科学实验。

建议实际运行方式和依赖见 §13.5 与 `PARALLEL_EXECUTION.md`。这些是明确授权的工程调度，不需要每启动一个已就绪作业再征求许可。

---

## 5. Raw 检索、候选与路径合同：完全由本轮生成

### 5.1 Raw 的唯一分数

`raw(a,b)=z_a·z_b`，z为同一冻结Qwen合同的单位向量。第二跳使用z_e，绝不能复用z_q。无学习投影、无Teacher、无GT。

离线训练候选用分块exact原始内积生成：各edge anchor Top256 hard reservoir；每q构造Direct100、20text+20image、每e Top20target。QT-only Student 的独立列表按 §9.7 只使用 raw QT 候选，不使用此两路列表。每次exact排序统一为score降序、UTF-8 object ID升序。不要按ID排序的集合截前20充当排名。

正式在线评测用本轮建的HNSW，M32、efConstruction200、efSearch=max(200,请求检索深度)、固定seed20260920、建索引和搜索num_threads=1。请求超合法可用数则取全部，不重复填充。

HNSW内部距离是实现细节，必须恢复成inner-product再报告。收到ANN候选后可计算同公式精确分数和稳定tie-break，不能偷偷用另一个模型重排来“验证索引”。

自表等合法性过滤后数量不足，检索深度按 `max(K+32,2K)` 开始、逐次翻倍至全库；每次记录实际请求。该规则对raw/所有Student相同，不能为某个方法单独提高efSearch。

### 5.2 本版候选准入公式（不读取旧校准映射）

每个q的自然实际到达路径为(q,e,t)，路径提议分数：
\[
a(q,e,t)=s(q,e)+s_{next}(q,e,t).
\]
Raw：`s_next=raw(e,t)`；C1 Student：无条件ET；C2：条件化ET。每个target按a降序、e ID升序保留最多4条**不同e的自然到达路径**。同一(q,e,t)重复记录去重，不让重复文件增加票数；所有别名/重复内容处理先写入内容映射，不用向量相近冒充内容相同。

本版只对完全相同的模型输入内容e做去重：canonical e为其模态内UTF-8最小ID；GT别名映射同步；不同source但模型内容完全相同也只做一次查询。table ID不合并，保持原target GT空间。内容key用实际encoder输入规范序列及图像像素内容指纹，不只用URL。

Evidence target分数为retained路径a的LSE；没有路径的target不出现在Evidence名单。Direct名单是Direct100。
\[
RRF(t)=\mathbb1[t\in D]/(60+rank_D(t))+\mathbb1[t\in E]/(60+rank_E(t)).
\]
U=D∪E；按RRF降序、target ID升序得C100。**等权且固定**，不是weighted RRF；不设置信度gate、手工quota、online implicit分流。Evidence-only target不能伪造direct票。

这份公式替代一切模糊的“沿用production默认值”。原生产代码只有在行为完全符合本节且不读取历史calibration时可复用。此次是新配方，不宣称与历史D1 coverage/其他准入版本完全相同；历史对照要重新按本节评测并标注。

### 5.3 哪些是“自然路径”

只有该q实际第一跳召回e、且相应第二跳把t召回的三元组才是自然路径。Teacher不允许在在线阶段额外检索全湖、扫所有GT witness或把全query evidence自动配给每张表。训练增强的共享witness是明确例外，须带`training_only=true`，绝不进自然评测。

原始arrival记录完整保存；retained≤4只限制昂贵Teacher评估的path数，不删掉审计所需的原始arrival。路径数不足不用空图像/零向量补足。

### 5.4 全正例保护与候选采样

所有新edge列表含全部该anchor已知正例，再取32个hard和32个均匀无放回random competitors。hard来自raw Top256，先按作用域移除P/I，再取前32。random从完整合法剩余库排除P/I/hard后取32；hard不足时把不足数量移入random。总竞争项最多64，合法库不够则取全部并记录。

**列表长度=正例数+最多64，不是固定总长64。**不能为满足batch shape裁掉多正例。随机采样用局部PRNG，见§10.3；不得对每个query的整个225K库逐个SHA排序。

### 5.5 一次且仅一次Teacher难例刷新

T_EDGE结束，用本轮T_EDGE在每个raw edge Top256 reservoir内打分，更新32hard；random部分按预定namespace重建，不读取dev。对于QT，hard reservoir使用raw Direct256与raw U的并集；这是真并集，不能把Direct列表拼自己一次冒充两路候选。

记录每列表刷新前后成员和score来源；刷新成员完全未改变时照实记录，不声称已经获得新难例。仍使用相同P/I保护。Teacher只能在这些有限候选池交互评分，不能全湖交互。

T_PATH每q训练目标集：`C_raw_exact100[q] ∪ G[q] ∪ top32_T_EDGE_QT_nonpositive(raw_U[q])`。包含自然两路竞争和所有正例；不裁成100。该训练graph之后供所有Student C2同条件使用；不得读取旧path graph，也不得由某个对照臂生成其他臂必须依赖的图。

---

## 6. 一种新 Teacher 架构：局部对象池化 + 共享交互 + 一个 head

### 6.1 固定结构

输入4096；modality adapter三类各Linear(4096,512)；table每schema/row一个token，额外区分schema/row kind；text/image使用新learned query pooler，分别16/24 latents、8heads、attention dropout=0、residual LayerNorm。输入为§4固定底座摘要，而不是旧Teacher latent。

Relation Transformer：3层、512维、8heads、FFN2048、GELU、pre-norm、dropout0.1，输出LayerNorm；没有全局绝对position embedding。三个role是source/query、target、bridge。type-pair embedding为有序3×3关系。

保留原架构的全局z信息分支，但它的所有参数也新初始化：
\[
g_x=LN_\tau(A_\tau z_x),\quad g_x\in\mathbb R^{512},
\]
\[
G(a,b)=W_o\,GELU(W_i[g_a;g_b;g_a\odot g_b;|g_a-g_b|;e_{\tau(a),\tau(b)}]+b_i)+b_o.
\]
W_i:2560→512，W_o:512→512。唯一head H为512→512→1，GELU、dropout0.1。

Pair：`[REL+type_pair(a,b)], C_a+modality+source_role, [SEP], C_b+modality+target_role`。

QET：`[REL+type_pair(table,table)], C_q+modality+source_role, [SEP], [g_e;C_e]+modality(e)+bridge_role, [SEP], C_t+modality+target_role`。

\[
f(a,\varnothing,b)=H(F(X_{ab})_{REL}+G(a,b)),
\]
\[
f(q,e,t)=H(F(X_{qet})_{REL}+G(q,t)).
\]
非空e的一条路径单独前向；不把不同e无分组拼成固定8槽，也不再将QE、ET两个pair标量相加冒充Teacher QET。

### 6.2 初始化与训练范围

所有Linear（含MHA投影）Xavier-uniform，bias0；LayerNorm weight1/bias0；REL/SEP/type/modality/role/kind/pooler query normal(std0.02)。每层独立初始化，不能Transformer克隆构造后留下三层完全同一初始权重。种子按§10.3。

T_EDGE与T_PATH均训练全部Teacher任务参数，包括modality adapters、poolers、global adapters/norms、Transformer/head。**不能照搬旧局部计划，把随机初始化的pooler或global adapter冻结。**唯一冻结底座是Qwen。

T_PATH 与 T_QT 分别从本run同seed的 T_EDGE 终点精确复制全部权重，各自换空AdamW。二者互不作为对方parent。不是外部warm start；分支各自完成后才允许缓存该分支可学习压缩器的C_x/g_x，键必须含分支及最终Teacher哈希。

## 7. Teacher训练损失与完整日程

### 7.1 通用多正例排名损失

对合法的positive集合P与competitor集合N：
\[
L_{rank}(s;P,N)=LSE_{i\in P\cup N}(s_i)-LSE_{i\in P}(s_i).
\]
ignore完全移除。P空或N空则该任务inactive，不返回伪0后参加该任务平均；不把空P改成G。该loss优化正集总质量，不保证每个正例都高，报告不夸大。

本轮统一保留这一定义，不同时换每正例loss、BCE、margin-ranking、KL方向等，以免配方再次膨胀。所有loss直接用raw logits，温度1，不能接10×sigmoid。

### 7.2 T_EDGE：完整原train两epoch

对每个q建立§3.4的五关系组。每关系先平均其active列表loss，再对该q的active关系等权平均，得到L_edge(q)。每个原train q有QT，应至少一个active项；若所有候选均正导致无competitor，记录inactive，但不得伪造负例。

每epoch全部active q按固定局部PRNG顺序一次；logical batch8个q，query microbatch1。每q所有候选的logits必须先合并再算同一个listwise loss，不能分块CE。两epoch后固定`T_EDGE/epoch2.pt`，不按dev选择epoch1。

完成§5.5的唯一hard刷新；输出actual member差异，不新增第二轮。

### 7.3 T_PATH：真实target-path两epoch

每q使用§5.5固定目标集合。Natural view为raw真实retained paths；Augmented view每epoch在原W[q,*]中按固定循环挑一个(e*,t*)，**向该q每一个候选t都添加同一条e***。每个(q,e,t)去重，已有e不重复投票，不删除原自然路径；最多4自然+1增强。无W则只有Natural。

选择anchor的来源为当前q原GT，不按模型分数挑容易正例。不得只给正t添加e*；新e导致的所有额外候选路径都要真实前向。

每t始终有一个同模型零跳候选：
\[
S_T^v(q,t)=LSE\bigl(f(q,\varnothing,t),\{f(q,e,t):e\in B^v(q,t)\}\bigr).
\]
无e时S_T=f0。这里的f0是同一head的0-hop候选，不是历史T0 QT补分。

**target discovery loss的正集固定为G[q]，不是仅取当前view看到了witness的子集。**阶段一目标是排序合法target；是否已经完整恢复连接由Stage2检查。缺少已标witness不把真实target标负或从GT分母删除。此规则避免把旧CLEAN-R1“当前支持J”和“全部目标Recall”无意混成两套冲突标签。

路径条件loss单独使用原W对应的P[q,e]/I/N，正target不代表任意e正路径。候选集为该e的raw ET Top256中取64hard +32random+全部P，按§3 mask。该小集合只用于Teacher交互与后续KD，不能代替Student C2的全目标SUP分母。

定义：
\[
L_T(q)=\overline{L_{target}(S_T^v;G[q],C_q\setminus G[q])}^{\,views}
+0.5L_{edge,new}(q)+0.5\overline{L_{cond}(q,e)}^{\,e\in W_q}+0.2L_{support}(q).
\]
新edge replay使用唯一刷新后的列表；同一Teacher和head处理，不另训第二Teacher。某组无active项该项为0但不改变其他系数；view均值只平均存在且active的view。

support仅针对原标注明确缺必要属性、t*∉D[q]的anchor：
\[
L_{support}=softplus(1+f(q,\varnothing,t^*)-f(q,e^*,t^*)).
\]
它要求正确witness提高相对支持，不把t*改成负target，不把未标注错误e当确认负例。不额外引入支持head。

**这些监督仍可能学出忽略E的捷径；必须通过最终E内容对照判断，不保证公式写了E就已利用E。**

### 7.4 Teacher固定优化与资源

Teacher edge/path及独立QT分支均lr5e-5，AdamW betas(0.9,0.999)、eps1e-8、wd0.01，clip全局norm1.0；无scheduler/warmup。阶段间reset optimizer，epoch间连续。logical batch8，microbatch1q；路径block8/4/2/1在正式训练前依同一内存试跑规则选最大能容纳值，之后固定。

允许对每个路径块做activation checkpoint并保持dropout RNG重放。必须跨块收集完整target/path logits并算全局loss，不能8路径一CE、8目标一CE。梯度累计的分母是本logical batch真实active query数，不是固定8，也不是各microbatch均值再平均。

每阶段保存init、各epoch末模型/optimizer/RNG/顺序游标与训练计数。固定T_PATH epoch2为最终T_frozen；即使epoch1更高也不替换。中间dev只是观察，不改变下一阶段权重或配方。


### 7.5 必须新增：独立训练的 T_QT 对照

**T_QT 不是 T_PATH 的 noE 评测，也不是载入历史 T0。**T_EDGE 2 epoch结束后，T_QT/T_PATH从相同全部tensor分叉、各自fresh AdamW、各2 epoch，target query顺序、目标候选IDs、G与target权重一致，均固定第二epoch末为终点。可两卡并行，不能先跑完T_PATH再把其权重拷给T_QT。

T_QT/T_PATH每epoch的query顺序均取既定T_PATH数据namespace；不能另用QT名称生成不同顺序。两分支各自初始化训练RNG，不共享全局随机状态。

T_QT的target scorer始终为 `r_QT(q,t)=f_TQT(q,EMPTY,t)`；该target前向不加载E、不拼接bridge token、不乘路径数量。正确target集合仍是**全部G[q]**，不是D[q]；不能把implicit正目标标负以降低对照能力。

本轮目标集合与T_PATH完全相同，见 §5.5。这意味着候选IDs可能由raw evidence带入，但 T_QT 的目标评分输入没有E；这是“相同候选竞争下的QT-only重排对照”，不是“训练阶段完全从未使用多模态信息”。

训练损失固定为：
\[
L_{TQT}(q)=L_{rank}(f_{TQT}(q,\varnothing,C_q);G[q],C_q\setminus G[q])
+0.5L_{edge,new}^{TQT}(q).
\]

edge replay沿用相同五关系的刷新列表及相同query-balanced归约。**保留五关系pair warm-up/replay，是为了保留已有强pair Teacher的任务组织；QT-only限定的是target重排及不使用QET联合路径训练。**T_QT不计算conditional QET loss、support margin、target-path loss或Natural/Augmented两次target loss。Natural/augmented在没有E输入时不产生两份重复target监督。所有可训练参数及优化器数值与T_PATH相同；未在T_QT目标前向使用的bridge参数没有该项梯度，不伪造更新。

T_EDGE末期的QT读出也作为零额外训练的 `T_EDGE-QT` 参考保存，用来观察QT fine-tuning是否帮助，而不是挑T_EDGE/T_QT中dev更高者作为正式T_QT。

**公平性与结论边界：**相同起点、目标候选、target标签、epoch、query更新量，但loss结构与前向FLOPs不相同；因此T_PATH−T_QT是两个完整训练配方的差异。T_PATH−其自身f0及E-swap是输入内容干预。不能把一种差异解释成另一种。

主线Student仍由T_PATH蒸馏，**不因为T_QT效果较好就把主线Teacher换掉**。T_QT只用于明确的对照和 §9.6/9.7 的指定Student监督，分支/score缓存严格隔离。

---

## 8. Student初始化与单向量公式

### 8.1 本轮重算PCA，不读取旧basis

只用原湖中不同target/evidence对象的z，不读任何相关性标签，不使用dev/test query对象。对完全相同的encoder内容只计一次；这是同湖无标签transductive预处理，不宣称inductive外部湖泛化。

float64分块累计均值和4096×4096 centered covariance，求前1024特征向量。每向量绝对值最大元素取正确定符号（并列最小坐标），降特征值排序；存eigenvalues、mean、W、对象ID/内容集合哈希和解释方差。使用确定性CPU `eigh`，不自动换随机PCA。

**本版采用“centered covariance求方向，Student应用Wz不减mean”的明确初始化约定**，不是同时声称做了标准centered PCA transform。W的row应正交；三种P均复制W，之后独立参数。无basis历史文件输入。不同seed共享这个纯数据计算产物，不共享任务训练权重。

### 8.2 基础Student

三类P_tau：Linear(4096,1024,bias=False)，table_query和table_target共用P_table。五个有向full R分别QT、Q→text、Q→image、text→T、image→T，初始identity。其他未训练关系不实例化成可训练参数。

列向量写法：u_x=P_tau z_x，v_{a→b}=R_ab^T u_a，s(a,b)=v^T u_b。

代码统一行向量：`u = z @ P.T; v = u @ R; scores = v @ U_dest.T`。目标index向量只能是u_t，不再乘R；text/image共享同一个table目标index。C1中所有P/R训练，条件化主线C2全部冻结；KD-NATIVE/QT-only对照的训练范围在 §9.6/9.7 明确规定，不可套主线冻结规则。

z已经单位化，u/v不再次单位化。没有额外sigmoid、10倍变换、0.07温度、模态标定或learned confidence scale。

### 8.3 条件化adapter与E-only

\[
h(q,e)=[u_q;u_e;u_q\odot u_e;|u_q-u_e|],\quad d_\theta=W_2GELU(W_1h+b_1)+b_2.
\]
结构4096→256→1024，输出层W2/b2全0，W1 Xavier-uniform、b1=0，无dropout；text/image共用一个adapter。E-only输入为[0;u_e;0;0]，参数量/初始化/训练标签/候选完全同QE。

\[
b=0.5\|v_e\|_2,\quad\bar d=d\min(1,b/(\|d\|_2+10^{-12})),\quad v_{qe}=v_e+\bar d.
\]
这里v_e=u_e @ R_eT，是**本轮该Student C1终点**产生，不是B13。原v_e=0时残差允许上界0，该item条件梯度为0并记录，不暗换归一化/epsilon base。

正式第二跳s_next=v_qe·u_t。train/full exact/ANN/cache都调用同一函数。零初始化与本轮C1 ET严格一致；norm约束逐样本满足。约束合成前的残差，不能通过单位化最终向量冒充该约束。

## 9. Student完整C1→C2与六个明确终点

### 9.1 所有对照都拥有完整fresh来源

每seed：新Student初值S_init复制到S_SUP、S_KD，各自空optimizer。C1分别SUP和SUP+KD；两者原GT、edge列表、顺序、初始化和预算一致。

C1结束后，S_KD_C1复制到两个原条件化子阶段KD-QE/KD-EONLY，并按9.6另派生KD-NATIVE。前两个子阶段的adapter初值逐tensor相同，分别空optimizer。该共享prefix是同一预定fresh run的一部分，不是旧checkpoint依赖，也不声称EONLY拥有独立C1随机性。

原主线三个Student保持为：SUP-QE、KD-QE（预定主模型）、KD-EONLY。新增 KD-NATIVE、QT-SUP、QT-KD；其中前者从本轮KD-C1分叉，后两者从新table P/R初始化独立训练，细则见9.6/9.7。没有按结果从SUP换成主方法的规则。SUP/QE用于整体KD配方对照；QE/EONLY用于固定KD C1起点、同监督下的Q输入增量对照。

### 9.2 C1：全部原train一次遍历

使用§5.5本轮刷新后的edge列表和§3.4 query-balanced归约。SUP只L_edge，KD为L_edge + L_KD_edge，系数1。Teacher为同seed T_frozen，eval+detach；edge分数用pair f(a,EMPTY,b)，不拿QET输出蒸馏QE或QT。

\[
L_{KD}(s,t;A)=2^2\,KL(softmax(t_A/2)\|softmax(s_A/2)).
\]
A为相同candidate ID且排除ignore；Teacher和Student candidate order一致、positive mask一致。Teacher无梯度；每list先求KL再按和SUP相同的任务/query层次归约。不能Teacher→Student方向写反，不能先把每个query不同长列表拼一起softmax。

C1 lr1e-4、logical batch64q、microbatch4q、AdamW同§7.4其余设置；仅1epoch。所有P/R更新，adapter不参与也不更新。该完整一遍不是强制复制历史356/659步，updates=ceil(active_train_queries/64)。

每个主线SUP/KD分支完成自身C1后即可各自冻结P/R，建立本run自身target/text/image索引以及全目标矩阵。旧Student不能替代任何C1输出。

### 9.3 C2：每query聚合全目标条件监督与target-path监督

C2每epoch遍历原train q。其条件项包含**该q全部不同、有效、原GT验证的e**；不是每epoch随机只取一个来凑旧pair数。每个e的正集为P[q,e]，ignore按§3.3。

完整target SUP使用该Student本轮C1冻结的U_T，对全部legal targets计算：
\[
L_{full}(q,e)=LSE_{P\cup N}(v_{qe}^TU_T)-LSE_P(v_{qe}^TU_T).
\]
分母不是小KD列表；P/I的遮罩完整覆盖整个合法库。忽略项即使也是另一个Q的positive也不能误标负。

条件KD使用§7.3那个共同的小QET列表（64raw-hard+32random+全部P），Teacher读同一q/e/t，Student读同一q/e，mask一致。无历史Qwen Teacher-QT替代QET。full SUP和small-list KD是两个项，不宣称Teacher全湖交互。

C2 target训练固定同§5.5的raw训练graph和全部G；不跟随某个Student臂改graph，避免KD/SUP对照拥有不同候选组成。Natural/Augmented两view与Teacher阶段同规则，anchor随本C2 epoch循环重新确定；没有GT增补进入dev/test。

Student路径分数：
\[
p_S(q,e,t)=s_S(q,e)+s_{next,S}(q,e,t),\quad S_S(q,t)=LSE(s_S(q,t),\{p_S(q,e,t)\}_{e\in B(q,t)}).
\]
注意第一跳在这里是**数值常数但参与完整路径分数**，C2不对它反传。Teacher对应S_T在同一candidate/path集合上计算，零跳和QET均为自身函数，不用Student提议分数做Teacher评分。

\[
L_{C2}(q)=\overline{L_{full}(q,e)}_e + \lambda_{KD}\overline{L_{KD,cond}(q,e)}_e
+0.5\overline{L_{target}(S_S^v;G,N)}_v
+0.5\lambda_{KD}\overline{L_{KD,target}(S_S^v,S_T^v)}_v.
\]
SUP的lambda_KD=0；KD-QE/KD-EONLY为1。无有效e时条件项为0，仍保留target项，不把无witness query从数据集中删掉。无自然/增强e导致所有目标项只依赖冻结零跳时，该q loss是常量，记录`zero_trainable_path`，不能伪造可训练梯度或使optimizer多走一步。logical batch按计算图上有adapter路径的q构成；名单在该epoch数据准备期按结构条件确定，与模型预测无关，三臂共用。不得根据实际梯度恰为0、loss大小或v_e范数删除q；这里排除的仅是该q没有任何可训练路径、所有项都只依赖冻结常量的结构情况。

**C2只有adapter可训练，所有P/R和index固定。**不可为“帮助拟合”悄悄解冻QT/P，不能把path训练只实现成若干QET pairloss后漏掉target项。

C2 lr1e-4、logical batch64q、query microbatch1、3epoch，AdamW与clip同前。每epochwitness量不同不改变q权重；同q条件e取平均、views取平均。C1/C2 optimizer必须重置，C2 epoch之间连续。

### 9.4 教师缓存与蒸馏真实性

T_frozen之后，可缓存本轮需要对象的learned C/g和本轮训练列表的Teacher logits；键包含Teacher哈希、完整q/e/t、ordered candidate IDs、mask、view和协议。缓存QET时必须含q，不能只用e。不能把T_EDGE/旧T0分数读成T_frozen。

lambda_KD=0时SUP的loss、梯度与完全不运行KD的代码一致，且不为省事把Teacher分数作为SUP标签。记录KD项实际非零梯度贡献，不以“参数中写了1”证明KD生效。

### 9.5 主模型与中间观察

SUP-QE、KD-QE、KD-EONLY正式终点为各自C2 epoch3；QT-only亦为其C2 epoch3，KD-NATIVE为其C2 epoch1。不选较高的中间epoch，也不将C1改名final。每epoch保存状态和有限健康指标，首个seed不因中间Recall未涨而改变流程；数值错误/冻结约束破坏立即INVALID_RUN，不用旧parent补回。


### 9.6 新增 KD-NATIVE：真正没有条件模块的旧式 Student 结构

目的：为“原P/R双线性结构、不让E→T读取Q、最终配合QT重排”保留可独立追溯的fresh工作点。**它仍使用evidence，不是Direct-only；KD-EONLY带有新MLP，不能代替这一组。**

起点：本run同seed的 `S_KD_C1/epoch1`，与KD-QE/KD-EONLY共享相同fresh前缀。保存其 `NATIVE-C1` 读出作为零额外训练的中间参考，但正式KD-NATIVE还要完成下面C2。该C1已经用本轮T_PATH的pair输出蒸馏；必须披露，不声称它全谱系只受T_QT监督。

结构：仅原三类P与五个full R，完全不实例化/加载/训练conditional adapter。所有E→T打分严格为
\[
s_N(e,t)=(P_ez_e)^\top R_{eT}(P_tz_t),\quad
p_N(q,e,t)=s_N(q,e)+s_N(e,t).
\]
固定e/t时改变q不能改变第二跳分数；第一跳或最终路径分数随q变化是合法的。

C2训练全部P/R，1 epoch完整原train、logical batch64q、microbatch1q、lr1e-4、wd0.01、clip1、fresh AdamW，其余数值与主线相同。没有rho、没有全目标conditional loss。**目标向量在此C2训练中会更新**，不得拿C1的旧key打分；在C2末重新构建own索引。在线索引仍是每target一个静态向量，不是训练期间永不更新。

使用与主线相同的 §5.5 固定target集合、自然/共享增强路径和G。Student聚合为
\[
S_N^v=LSE\bigl(s_N(q,t),\{s_N(q,e)+s_N(e,t)\}_{e\in B^v}\bigr).
\]
监督Teacher使用**本轮T_QT的pair输出**构造旧式分数：
\[
S_{pair,TQT}^v=LSE\bigl(f_{TQT}(q,\varnothing,t),
\{f_{TQT}(q,\varnothing,e)+f_{TQT}(e,\varnothing,t)\}_{e\in B^v}\bigr).
\]
不调用T_QT的QET前向，不接T_PATH的QET logits，不读历史native cache。蒸馏温度2、系数1，loss固定为
\[
L_N(q)=\tfrac12\left[L_{rank}(s_N(q,C_q);G,N)+L_{KD}(s_N(q,C_q),f_{TQT}(q,\varnothing,C_q))\right]
+\tfrac12\overline{\left[L_{rank}(S_N^v;G,N)+L_{KD}(S_N^v,S_{pair,TQT}^v)\right]}_v.
\]
P/N使用target作用域G/C-G，所有正例保留；无e时两部分都退化成合法QT项，不删q。没有额外edge replay、anchor或Uniform。条件化分支使用的P/I/N不误套到目标级G标签。

这是**旧式结构的明确新训练配方对照，不是历史B13精确复现**：新数据范围、Teacher、raw-logit损失、学习率、候选及C2预算均已写清。KD-NATIVE与KD-QE的差异包含C2训练范围/监督Teacher/预算，不能称仅“加Q”单因素；后者由KD-QE−KD-EONLY回答。本组主要系统为 `KD-NATIVE own两路C100 + T_QT`，另在相同C100/自然路径上给T_PATH排序，拆开retriever与reranker效果。

### 9.7 新增 QT-SUP 与 QT-KD：独立训练、不输入 evidence 的 Student

这两组回答“只使用表格query/target的检索训练和Direct ANN，能做到多少”。**不能从已训练的多模态Student关闭E入口后冒充独立训练组**；那是 §12.1 的纯推理消融，另列。

**初始化/模型：**只实例化 `P_table:4096→1024` 和 `R_QT:1024×1024`，P取本轮相同PCA，R为identity，两臂初值相同。无P_text/P_image、QE/ET关系、adapter或路径聚合。共享PCA本身按§8使用无标签多模态湖数据，因此“无E”精确指当前任务的模型输入、训练样本/监督项及检索不读E；不宣称公共底座/PCA从未接触多模态内容。

**允许监督：**原train的G和合法target规则；不读D/W/Epos、evidence内容/embedding、已标witness、Path teacher logits、两路候选。QT-KD额外读取T_QT在相同Q/T上的f0。T_QT的五关系warm-up是其离线来源，因此QT-KD不是“全谱系绝无多模态辅助信息”；QT-SUP没有此蒸馏来源。报告必须同时列二者。

**QT-C1（各1 epoch）：**全部原train active q，logical batch64、microbatch4、lr1e-4。列表为全部G + raw QT Top256中排除G后32hard + 全legal target排除G/hard后32random；不足hard从random补齐，列表ID/顺序两臂完全相同，不用raw U。loss分别为 `L_rank` 与 `L_rank+L_KD(T_QT f0)`，KD温度2、权重1。

**QT-C2（各3 epoch）：**各自继承本轮QT-C1，fresh AdamW、同lr/wd/clip、logical batch64、microbatch1；继续训练全部table P/R，**不是冻结P/R后训练不存在的adapter**。主SUP分母为完整合法目标库，P=G、I为空、N=legal-G；目标keys必须由当前P计算且允许P梯度，不能复用旧step的投影结果、detach keys或用只适合冻结目标的分块函数冒充。

QT-KD的小KD列表：全部G + T_QT在raw QT Top256非正候选上排出的64hard +32random（不足依规则补），每q固定一次，QT-SUP使用同一份元数据但不运行KD。只用于KD，不替代完整SUP分母。不是T_QT全湖交互，不做Student反馈刷新。每q损失分别为
\[
L_{QT,SUP}(q)=L_{full,QT}(q),\qquad
L_{QT,KD}(q)=L_{full,QT}(q)+L_{KD}(s_{QT}(q,C_q^{KD}),f_{TQT}(q,\varnothing,C_q^{KD})).
\]

§10的跨chunk全分母、尾batch和query均值规则仍适用；新增单测必须核对query和target两侧P梯度。QT-C2没有path loss、QET loss、rho、support或按implicit筛样本。各自固定C2 epoch3终点。

**在线：**只运行该模型自己的Direct100；`C100=Direct100`，不跑Q→E/E→T，不借其他模型U，不做RRF；Student得分排序，另由T_QT f0重排这同一Direct100。严格Direct-only行不能为了对照T_PATH而临时取E。Implicit正目标仍是G中的正常目标，不因无E被标负或丢分母。

**匹配对照说明：**QT-SUP/QT-KD在各自C1/C2训练人口、候选、初值、逻辑更新数一致，只差KD。与完整多模态Student是不同训练配方；区分“独立无E训练”和“同一多模态模型只做Direct推理”，二者都要报告，不能互相替代。

---

## 10. 分块、采样、训练计数：禁止隐藏更换问题

### 10.1 全目标分母分块的数学要求

默认target chunk4096；若显存试跑不能容纳，可在正式训练前依序2048/1024/512/256选最大能容纳值。chunk只是计算方式，不能在每块内独立softmax或独立loss再平均。

对每item各chunk计算valid全部logsumexp A_k、positive logsumexp B_k，流式合并A=logaddexp_k A_k、B=logaddexp_k B_k，loss=A−B。P跨chunk必须完整；没有positive的某块B_k在数学上为−∞；实现时跳过空贡献/用None初始化累积项，不把`logaddexp(-inf,-inf)`放入autograd图（其反向可能非有限）。NaN/empty处理必须保留正确梯度。允许两遍重算/激活checkpoint，需与单矩阵损失、参数梯度一致。

目标向量仅在SUP-QE/KD-QE/KD-EONLY的条件化C2冻结。KD-NATIVE与QT-only C2会训练P/R，必须使用当前参数的target keys，完整P梯度不能detach；它们的最终索引在训练后创建。一次矩阵和每步更新后的query向量匹配。C1的候选投影可训练，不允许把epoch初期的旧key当本step key；C1训练只用当前参数对候选编码。

### 10.2 批量loss与尾batch

累积microbatch的每个q loss除以这个logical batch的真实active q数；累积完才clip/step。最后不足64/8的batch用真实数。所有微batch梯度先归约再裁剪；不能对每个小块分别clip改变算法。

Teacher phase / Student C1 / Student C2 的active定义在对应章节；输出每epoch数据项、query、列表、条件pair、零梯度结构项、optimizer steps分别计数，不能把这些数字混叫sample。Nepoch变化时按ceil重算，不裁/重复凑历史342步。

### 10.3 随机性

所有对象/anchor初始先UTF-8 ID排序。每个随机用途仅对namespace做一次SHA256，前8bytes big-endian转整型初始化独立`random.Random`。无放回采样用预索引合法库和排除集合拒绝抽样/列表sample；大库每query不逐对象哈希。

namespace前缀 `FRESH-PATH-v2|20260920`。包含任务、model_seed（若适用）、phase、epoch、q/e ID、purpose。edge原始候选及raw图不含模型seed；Teacher刷新含当前T seed；新增对照的数据namespace使用独立CONTROL前缀且不消耗主线RNG；paired Student使用相同data namespace，不写SUP/KD。model初始化、data sampler、bootstrap三个RNG互不消耗。

Teacher path/C2的anchor：原q全部(e,t)标注去重排序，offset由`anchor|q` namespace产生，epoch r取 `(offset+r−1)%len`；同一query不同Student臂/Teacher同阶段用同anchor。phase进入namespace，不能误读Teacher epoch数作为C2 epoch数。

启用确定性算法、TF32关闭、记录CUDA确定性配置。若某算子不支持，可使用数学等价deterministic实现并通过梯度差分；不能quiet warn_only。不要为了位级跨硬件一致声称可证明最终结果完全相同。

### 10.4 预取与优化

预取只能提前做当前固定stage/epoch的无梯度CPU准备；按序消费，不按worker完成先后训练。compact→expand必须完整保存空positive_ids、ignore、q/e、view、候选顺序和权重。禁止在GPU算完后发现无效样本再更改batch分母。

允许相同本轮list的Teacher logits缓存、同一次forward中重复对象投影共享、日志异步写；不允许缓存trainable压缩输出跨optimizer step，不允许把不同arm的已训练参数/候选向量缓存混用。

## 11. 主检索与真正的Path重排

### 11.1 每个Student自己的候选

主线条件化终点从自己的C1 P/R和C2 adapter、KD-NATIVE从自己的C2终点P/R，完整运行Direct100、20text+20image、每e20target，按§5准入；不是在旧U/C100中重新排列，也不是只重评共享raw候选然后称own retrieval。QT-SUP/QT-KD只运行自己的Direct，严格按§9.7，不套本段两路检索。

相同C1的KD-QE与KD-EONLY必须共享完全相同Direct、第一跳、目标向量及HNSW实例；只有第二跳query向量变化。SUP拥有自己的C1，所以不得强行复制KD的Direct/firsthop。

模型query cache包含model hash、q/e/source/destination类型；目标index hash仅取目标空间参数/ID/建索引配置，不因adapter变化强制按Q重建。

### 11.2 最终评分

对主方法的own C100每t使用同seed T_PATH（原T_frozen角色），读取真实自然retained e：
\[
R_T(q,t)=LSE(f_T(q,\varnothing,t),\{f_T(q,e,t)\}_{e\in B^{natural}(q,t)}).
\]
不加Student score、RRF score、外部QT、列概率、GT支持数。C100内每t至少有零跳评分；不能因没有e丢t。所有方法的模型输入禁用candidate ID/排名/来源标签。

共享的G(q,t)内部global表征不等于外部QT标量融合，但它存在绕过E的风险。最终E-swap/noE必须报告；真实E不产生价值时如实判主张不足，不能按更漂亮的QT-only成绩改论文主方法。

### 11.3 不引入在线标注路由

implicit/explicit仅用于train监督定义和report切片。对全部query相同检索、零跳、QET、聚合；不能知道implicit就关f0、知道explicit就跳过e。未发现已标witness不允许GT补e。

### 11.4 历史参照隔离

训练与dev决策冻结后，单独命令`compare-history`可以读取用户实际提供的B13/当前最强Student/T0进行只读重评，原始数据/当前候选规则一致，明确它们是historical lineage。缺历史文件时标REFERENCE_UNAVAILABLE，不阻断fresh完成。

必须预留同协议历史表位；不能把历史47%直接复制到本轮不同cache/候选规则的主表。只有适配合同确实相同才给差值；否则单列“历史原协议报告”，不能推断因果。历史参照进程的输入路径和输出不进入任何训练manifest、seed重复条件或checkpoint选择。

---

## 12. 有限而完整的评测合同

### 12.1 主表与候选表（所有原dev query）

固定报告overall / implicit / explicit，query-macro Recall@10、@20、@50（CR@50），以及每query分子/分母。主终点是KD-QE自己的C100+T_frozen Path R@10。

必须输出以下系统行。标为“同池”时candidate IDs逐项相同；标为“own”时各自真实检索，不能伪造公平候选一致。

| 系统行 | 候选与E | 最终分数 | 回答的问题 |
|---|---|---|---|
| Raw Direct | own D100，无E | raw QT | 无训练Direct基线 |
| Raw两路 | own C100/paths | admission原序 | 无训练两路基线 |
| Raw+T_QT / Raw+T_PATH | 相同raw C100；QT不读E，Path读自然E | 各自QT / Path | 同原始候选下的两种Teacher |
| QT-SUP / QT-KD Direct | 各自D100，无E | 各自Student QT | 独立无E训练的Student |
| QT-SUP+T_QT / QT-KD+T_QT | 各自D100，无E | T_QT f0 | 独立Direct-only完整系统 |
| KD-NATIVE+T_QT | own两路C100；Teacher不读E | T_QT f0 | 旧式结构+QT重排的fresh工作点 |
| KD-NATIVE+T_PATH | 上一行同一C100及自然paths | T_PATH Path | 固定旧式retriever时的Path效果 |
| SUP-QE / KD-EONLY / KD-QE + T_QT | 每模型own两路C100，QT不读E | T_QT f0 | 固定QT Teacher比较检索器 |
| SUP-QE / KD-EONLY / KD-QE + T_PATH | 同上每模型同一C100/自然paths | T_PATH Path | 主线和原SUP/Q条件消融；KD-QE为主方法 |
| NATIVE-C1+T_QT | 当前KD-C1的own两路C100 | T_QT f0 | 中间旧式读出；不是正式KD-NATIVE终点 |

所有Student另报自身Direct、Evidence、U覆盖、C100覆盖；多模态模型报Student QT与Path在同一U/C100上的排序。QT-only没有Evidence名单，U=D，任何Evidence指标写N/A而非伪造空E成功率。集合按ID存储不能取lexical前10当排名。

**同一个模型的推理期无E消融：**至少对KD-QE和KD-NATIVE，在同一冻结模型下比较 `own D100+T_QT` 与 `own C100+T_QT`。前者不执行E检索，后者只让E增加/改变候选；reranker相同。这一差异测候选发现收益，不测Teacher是否读懂E，也不是独立无E训练Student。

对每个两路模型查询其Direct Top `|U_q|` 得MatchedDirectM，报候选覆盖及Student QT排名；另以固定T_QT比较D100/C100时两者均100个候选。MatchedDirectM大池不得与C100声明相同Teacher成本；本轮不要求大池Teacher全重排。所有行均保留overall/implicit/explicit和失败分母。

### 12.2 指标定义

\[
Recall_q@K=|TopK(q)\cap G_q|/|G_q|,\quad R@K=\frac1{|Q|}\sum_q Recall_q@K.
\]
RawUnionRecall/CR100将TopK替换为集合U/C100。每query正例多个时不能用Hit替代Recall。候选Top10理想上限为 `min(10,|C∩G|)/|G|`，不误称C100覆盖就是可达R10。

G空的query按照数据集已明示任务定义固定排除并单独列出；不能因为模型未命中或输入读取失败而排除。准备期缺必须特征应先修复/阻断，最终意外推理失败保留该query原分母计0，并报告failure count。

### 12.3 条件化第二跳的固定probe

在dev中构造所有 `e∈raw第一跳(q)∩W(q,*)` 的不同(q,e)，固定给所有模型。P是该q/e的评测标注正目标。按各模型自己的静态target空间做full-lake exact及相同HNSW搜索，报告ET R@10/20、MRR、正例rank中位数、common top1占比。

先pair内算Recall，再q内平均其所有有效e，最后query-macro；text/image另切片。只含implicit或部分query时明确支持人口，不外推到全部dev。Real-Q与同模态固定Shuffled-Q只做最终推理对照，不能用shuffled标签训练负例。

### 12.4 strict evidence-only

每模型、每query定义：
\[
Z_{m,q}=G_q\cap E_{m,q}\setminus(D^{ANN100}_{m,q}\cup D^{exact100}_{m,q}).
\]
这是本模型实际新发现集合；另定义固定raw队列Z_raw以及共同C1队列Z_C1_KD用于跨adapter留存比较。保留ID，不复用历史207/213。统计其进入U/C100/最终Top10/50的数；分母为对应固定队列原数量，不能每阶段换分母。

同时报告严格直接预算外的所有标注目标 `G\(D_ANN∪D_exact)` 的query-macro Recall，和发现队列留存率区分。不能只挑含strict target的pair再把非strict positive一起算进去。

### 12.5 Teacher同池与E内容对照（必须包括独立训练的QT）

冻结KD-QE own C100、自然路径slot、q/t人口后，依次计算：

| 名称 | 具体权重/输入 | 是否独立训练 |
|---|---|---|
| Raw-QT | 原z_q·z_t | 无任务训练 |
| T_EDGE-QT | 本run T_EDGE epoch2，f0 | 只完成公共edge训练，中间参考 |
| T_QT | 本run T_QT epoch2，f0 | **独立完成QT target训练，主要QT对照** |
| T_PATH-f0 | 本run T_PATH epoch2，e=EMPTY | 同一Path权重的输入移除 |
| T_PATH-Real | 同一T_PATH，零跳+自然QET LSE | 主方法重排 |
| T_PATH-E-swap | 同一T_PATH，固定slot、替换E内容 | 内容干预 |

NoE与T_PATH-f0逐分数相同，只算一次。**严禁只报T_PATH-f0就写“已完成QT-only对照”。**另在Raw own C100重复T_QT/T_PATH-Real/T_PATH-f0，形成与主Student无关的共同候选环境；不按哪一池分数高选主表。

E-swap规则仍为：该split出现的canonical e按模态/内容key排序，在同模态不同内容间作固定非恒等循环置换；保持q/t、候选ID、slot数、模态和零跳不变。同一e在所有slot同donor，swap后不再次按donor去重。少于两个不同内容时该模态不可构造，不补零/跨模态。donor由内容规则选，不按GT挑“错误E”，不称确认负证据。缺donor特征按相同合同提前补齐。

报告 `T_PATH-Real−T_QT`（完整配方）、`T_PATH-Real−T_PATH-f0`（输入移除）、`Real−E-swap`（内容）三个不同差值、分项W/L/T与source bootstrap。另将同池Student raw QT、Student Path读出保存，不能把Teacher差异与候选入池差异混在一起。

E分支在T_QT训练的edge warm-up/replay中存在，但其target重排不读E；QT-KD亦有此间接监督来源。只有QT-SUP任务训练无E标签输入。上述scope如实写入 `CONTROL_SCOPES.json`，不把这些不同消融统称为“全流程移除多模态”。

### 12.6 统计

paired W/L/T以每query Recall差和tol1e-12计算。Bootstrap以原source group为抽样单元10000次、seed20260920；抽到一组时保留其全部query，以各query等权计算metric。多个seed先对同q聚合，不能当query人口翻倍；另外逐seed表必须保留。

开发集重复取舍明确标为exploratory，不把CI称未见测试集证据。Test仅在全部训练超参数、最终模型规则和seed数锁定后打开一次；结果差也不返工调配方。若历史test曾被开发查看，如实写“原test回归评测”，不能通过改名字/换seed宣称盲测。

### 12.7 效率

记录每阶段墙钟、峰值显存、累计optimizer updates、实际Qwen新增前向对象数、cache命中/新增字节。在线分别报Student ANN、候选准入、Teacher path前向次数与总时延p50/p95；固定GPU/CPU、batch、warmup次数。GPU计时用同步边界；正式单请求时延测量独占一张4090，并发吞吐另外测量。Raw、SUP、KD所用budget相同不等于FLOPs完全相同，真实path数另报。

---

## 13. 预算、止损与是否重复：不再无限诊断

### 13.1 每seed严格13个训练stage，不隐藏新增对照开销

| Stage job | 权重parent / 监督来源 | epoch | 训练参数 |
|---|---|---:|---|
| T_EDGE | 本轮新随机Teacher / 原train | 2 | 全Teacher |
| T_PATH | 本轮T_EDGE / path+edge | 2 | 全Teacher |
| T_QT | 本轮同一T_EDGE / QT+pair edge replay | 2 | 全Teacher（只对实际用到参数产生梯度） |
| S_SUP_C1 | 本轮PCA/identity / 五关系SUP | 1 | P/R |
| S_KD_C1 | 相同新初值 / SUP+T_PATH pair KD | 1 | P/R |
| S_SUP_QE_C2 | 本轮S_SUP_C1 / 主线SUP | 3 | adapter |
| S_KD_QE_C2 | 本轮S_KD_C1 / 主线SUP+KD | 3 | adapter |
| S_KD_EONLY_C2 | 同一本轮S_KD_C1 / 相同损失但Q置零 | 3 | adapter |
| S_KD_NATIVE_C2 | 同一本轮S_KD_C1 / T_QT pair-native监督 | 1 | 全P/R，无adapter |
| S_QT_SUP_C1 | 新table PCA/identity / QT SUP | 1 | P_table/R_QT |
| S_QT_KD_C1 | 同一新table初值 / QT SUP+T_QT KD | 1 | P_table/R_QT |
| S_QT_SUP_C2 | 自身QT-SUP-C1 / QT全目标SUP | 3 | P_table/R_QT |
| S_QT_KD_C2 | 自身QT-KD-C1 / QT全目标SUP+T_QT KD | 3 | P_table/R_QT |

每seed共2个最终Teacher分支、1个中间T_EDGE参考、6个正式Student终点；公共T_EDGE和公共KD-C1不重复训练。**最多2个seed、26个有效训练stage。**v2.0的7/14预算由本次新增对照明确改为13/26，不保留两个互相矛盾的计数。单纯同池多种读出、Direct推理消融、历史只读评测不算新增训练stage。

不额外增加Teacher bootstrap Student、第三Teacher分支、QT-only更多变体或超参数搜索。出现代码错误需重跑时，额外实际开销单独记录、受影响结果标invalid；不得把重跑隐藏在13个有效阶段中声称没有额外消耗。

### 13.2 必须立即停止的情况

NaN/Inf、标签边界泄漏、目标index意外变化、条件化C2非adapter参数更新、rho越界、对象输入不合规、真实候选/路径丢失、缓存键错误、资源无法满足：INVALID/BLOCKED。已经产生的结果保留但不可用于科学结论。不得转用QT-only或历史模型完成。

### 13.3 首个seed的性能止损在整链结束后

不设置“Teacher先超过历史T0才准训Student”的历史依赖gate，不设置“先在B13上adapter成功才准fresh”的gate。seed13主线和QT/旧式Student对照的全部13个合法阶段完成后出全部指定最终表；不根据中间loss低就宣布成功，也不为中间分数低追加epochs。

是否重复seed29由以下**全部**条件决定，数值单位为绝对Recall（0.005=0.5百分点）：

- KD-QE+T Path的overall R10 ≥ Raw+同T Path +0.005；implicit不低于Raw+同T。
- 同池T_PATH Path overall R10 ≥ 当前T_PATH f0-only −0.005，且 ≥ 独立训练T_QT −0.005。不能只选择较弱的QT对照。
- 同池implicit Real-E Path R10 ≥ E-swap +0.005。
- 固定probe ET R10(KD-QE) ≥ KD-EONLY +0.005。
- KD-QE+T overall R10 ≥ SUP-QE+T −0.005。
- 固定raw严格EO队列至少20个GT pair，KD-QE最终留存数量不低于KD-EONLY；不足20标证据不足、不重复，不把缺数据解释为方法已反证。

这些是资源取舍阈值，不是统计显著性宣言，也不是已证明最优的超参数。任一不满足→`COMPLETE_SEED13_STOP_NO_REPEAT`，锁定仅使用seed13的最终模型，完成既定一次test与报告后结束；不继续诊断新假说、不重训Teacher、不降低门槛。

seed29若启动，从新随机Teacher开始，重新做本seed公共Teacher刷新、T_PATH/T_QT分支与全部六个Student终点；只共享纯底座特征、本轮PCA和raw无标签候选。不能直接复制seed13 Teacher成为第二seed。两seed都结束后无第三seed。任何控制组都不根据seed13赢家单独决定是否复制；seed29若执行则完整13阶段，保持表格可比。

### 13.4 结果命名与科学结论

完整seed13跑完即可以标`FRESH_CHAIN_COMPLETED`，与性能是否超过raw/B13无关。`USEFUL_GAIN`、`Q_INCREMENT_SUPPORTED`、`KD_GAIN_SUPPORTED`、`E_CONTENT_USED`分别由相应表与CI判断，不能相互替代。缺B13参考不影响fresh身份。

KD弱但SUP好：报告KD主张不成立，不能把SUP改名KD；QT-only好但Path弱：报告路径主张不足，不能改主方法；只有probe上升但最终Recall不上升：不能称整链改进。


### 13.5 就绪依赖与建议并行顺序

完整依赖图以 `EXECUTION_DAG.json` 为准。权重parent与只读监督依赖是两种边，不能只检查checkpoint存在就提前运行。

1. P0/P1后T_EDGE运行；PCA、纯CPU候选准备、QT-SUP-C1在其各自输入就绪后可使用空闲另一卡。QT-SUP只依赖原QT列表，不等待T_PATH。
2. T_EDGE完成、hard刷新与共同target图封存后，T_PATH和T_QT分别从相同终点启动，可各占一张卡。S_SUP_C1只需刷新后的edge列表、无需Teacher logits，可作为合格共驻任务。
3. T_PATH完成后S_KD_C1可启动；T_QT完成后QT-KD-C1可启动，QT-SUP自己的C2无需等QT-KD。首选SUP与对应KD分配到不同卡，不能把每一条支线强制串行。
4. S_KD_C1完成后KD-QE与KD-EONLY从相同初值分叉，可双卡同时跑；KD-NATIVE还要等待T_QT完成。S_SUP_QE_C2只依赖T_EDGE后封存的共同graph及自身SUP-C1，不需要Teacher软分数；所有监督/缓存版本封存才启动。
5. 不存在的上游阶段不能通过加载历史同名文件补齐。已就绪任务用固定优先级队列调度；先主線关键依赖，再等待较久的控制任务，不按dev分数分配计算。
6. 正式dev各模型评测在该模型终点冻结后可提前执行，但 repeat/test 决策等待全部13阶段及规定表完成。正式效率计时安排独占窗口。

优先级和并发上限只影响墙钟，不决定数据/模型随机性。`EXECUTION_DAG.json`列出唯一13节点；不要求每seed顺序运行13个进程，完成独立阶段可以同时占用同一或不同卡。

---

## 14. 实现模块、执行命令与阶段验收

### 14.1 代码结构（职责明确，不做通用调度框架）

```text
src/fresh_path/
  config.py           # 解析本版JSON、拒绝旧profile及外部task parent
  inputs.py           # 原始dataset/split/内容展开；输入许可
  labels.py           # G/D/W/Qpos/Epos/P/I/N
  features.py         # 纯底座cache合同、按需生成、行/mask
  candidates.py       # raw精确池、统一采样、自然图/增强图
  teacher.py          # 单一fresh Teacher及pair/QET接口
  student.py          # PCA/五关系、受限adapter、统一ANN向量接口
  losses.py           # rank/full-denominator/KD/path及归约
  train_teacher.py    # T_EDGE→唯一刷新→独立T_PATH/T_QT
  train_student.py    # 条件化三终点、KD-NATIVE、QT-only两终点
  retrieval.py        # 本轮HNSW、Equal RRF、retention
  evaluate.py         # own/fixed/probe/swap/strict/统计
  lineage.py          # 当前run有限DAG、resume和只读对照边界
  scheduling.py       # 薄进程队列/资源预约，不建设通用平台
src/run_stage1_fresh_path.py  # 薄CLI
```

可以组织为少量现有模块而不逐字建所有文件，但职责/API与测试必须落到真实代码。不能先搭一套大量compliance抽象而迟迟不实现训练。脚本里不得import旧run_stage1_rXX的配置/路径解析作为默认入口；可提取其纯model/math函数到无数据副作用模块。

### 14.2 必须实现的CLI（以下是要求实现的接口，不声称服务器已有）

```bash
python src/run_stage1_fresh_path.py run \
  --dataset-root <原始20K数据集根目录> \
  --backbone-dir <公共Qwen本地目录> \
  --protocol <本包protocol.json> \
  --work-dir work/mmdd_stage1_fresh_path_v2_1_20260920 \
  --devices 0,1
```

参数中没有历史checkpoint或old train list。`--pure-feature-cache`可重复指定且可不指定；传入任何缓存须由合同验证，不据其索引选训练人口。

薄CLI子命令：`prepare-data`、`prepare-features`、`prepare-raw`、`smoke`、`train-teacher --seed --stage {edge,path,qt}`、`train-student --seed --stage <13节点中的Student阶段>`、`train-students --seed`、`profile-concurrency`、`evaluate-dev --seed`、`decide-repeat`、`evaluate-test`、`export`、`compare-history`。`run`按本文DAG与资源条件并行调度已授权步骤，不反复询问“是否继续”；仅真实缺关键输入/发生不合规时停止。

### 14.3 两个前置关口，而不是每写一份JSON就新加关卡

**P0 Data Ready**：root/split明确、GT重建、完整对象z可用、raw候选与PCA生成、冷启动/无旧产物依赖检查。缺局部Teacher特征按4章补齐，不能把原132漏项当永久阻断。

**P1 Implementation Ready**：附带参考数学测试 + 仓库实际函数测试 + 小型真实端到端smoke。smoke从原始对象构造自然/增强路径，经序列化预取、Teacher前反向、Student C1/C2前反向、索引查询、最终Path排序、指标输出。采用独立临时模型，最多每阶段2个logical updates，完成后丢弃，正式模型重新初始化。不能把这些临时训练产物当正式stage0。

P1通过后自动训练seed13主线及全部对照，只有13.2的真实安全条件可以中断。性能问题按13.3在整链末决策，不恢复旧A/B顺序。

### 14.4 真实集成检查的必要样本

至少含：一个explicit、有witness的implicit、同E不同Q、多个positive、空W、跨模态、缺自然path但GT目标存在、重复三元组、非单位非对称R、尾batch。实际数据不含某边界可用独立synthetic单测补，但不能捏造“真实数据已覆盖”。

仓库必须证明：

1. 删除/屏蔽所有旧任务文件，训练初始化和本轮manifest构造仍可执行。
2. Qwen纯缓存命中与本版raw内容重新构造一致；Query原行数=序列化行数=独立行摘要数。
3. P/I/N闭包正确，dev GT扰动不改变训练构造，空P经过prefetch不消失。
4. T_EDGE/T_PATH/T_QT实际使用的随机学习压缩器确实有梯度；T_frozen无梯度；Teacher QET真的读取e。
5. Student非对称R的train/exact/ANN统一；e第二跳不误用q向量。
6. 条件化C2的P/R不变、index不变、delta初始0、rho满足；Qmask是唯一EONLY差异；KD-NATIVE/QT-only按照其专章更新P/R，不共用此冻结断言。
7. 全分母分块与一次矩阵的loss/梯度一致，positive跨chunk不漏；microbatch与整batch一致。
8. natural不为None；shared增强e到所有t；无重复路径票；Teacher最后调用真实LSE而非QT函数。
9. KD分布候选/条件/mask一致，lambda0严格退化SUP；C2 target-path项实际反传adapter。
10. main C100真由两路生成，候选coverage/recall/strict口径可由原始ID复算。

上列是必须的正确性验证，不是10项额外科研实验。测试文件、命令和结果一次集中交付。附带reference只覆盖数学合同，不能代替第1/2/4/8等真实接线验证。

### 14.5 数值容差与禁止调整

reference float64 loss/gradient比较用1e-9；实际float32无dropout单函数默认atol1e-5、rtol1e-4；Teacher实算dropout需固定RNG并比较同一次前向或确定性eval。排名并列以ID处理，不要求不稳定torch.topk tie随机顺序一致。

同一实现重复运行自身存在差异时先记录范围，不能提高容差掩盖wrong formula。任何改变容差的请求必须明确绝对差、相对差和是否改变TopK，不能执行模型自行从1e-5放到0.1。

---

## 15. 交付目录、回执与禁止虚报

```text
work/mmdd_stage1_fresh_path_v2_1_20260920/
  ROOT_INPUTS.json / RESOLVED_PROTOCOL.json / SOURCE_LOCK.json
  DATA_SPLIT_REPORT.json / ENCODER_CONTRACT.json / FEATURE_COVERAGE.json
  INIT_LINEAGE.json / PHASE_STATUS.json / READ_AUDIT.jsonl
  CONTROL_SCOPES.json / RESOURCE_PROFILES.json / SCHEDULE_EVENTS.jsonl
  CONCURRENCY_PARITY.json / LEGACY_COMPARABILITY.json
  labels/          # 全部从本轮原train产生
  features/        # sidecar; 不复制旧缓存全集
  pca/             # 本轮重算
  raw/             # 本轮raw ranks/index/train graphs
  seed13/teacher/{edge,path,qt}/
  seed13/student/{sup_c1,kd_c1,sup_qe_c2,kd_qe_c2,kd_eonly_c2,kd_native_c2,qt_sup_c1,qt_kd_c1,qt_sup_c2,qt_kd_c2}/
  seed13/eval/{own,fixed,probe,swap,strict,bootstrap}/
  seed29/...       # 仅repeat gate通过时
  references/     # 与训练目录/输入隔离
  tests/receipts/
  RESULTS.zh-CN.md / METRICS.csv / DECISION.json / ERRORS.jsonl
  execution_source/  # 实际运行源码快照+diff，不是最后编辑但没运行的版本
```

每个训练回执必须同时写：planned/implemented/executed/evaluated、种子、起点、parent来源、本epoch活跃query/item/list、预算及实际steps、模块梯度摘要、损失分项、实际候选ID/视图指纹、时间与资源。样本审计无需每步保存巨大tensor，但全部消费的item IDs/order/hash须可恢复。

正式结果保留每query raw排名ID、每target所有实际自然/retained路径ID与score、GT整数分子/分母、分组与失败状态。打包清单不要包含自身hash，不要引用未打包模型权重却写“完整权重已提供”；可另给权重路径/大小/hash与是否随包。

最终训练checkpoint、optimizer、RNG须保留到验收；不得只存best.pt覆盖各epoch。所有候选graph/list都是本轮产物，生成器代码和root hash随文件记录。

### 15.1 最终报告必须回答

- 是否真完成root→T_EDGE→T_PATH→C1→C2→own ANN→Path？缺任何阶段不得称完整fresh。
- 相对raw+同Teacher，有无实际检索/最终R10增益？SUP/KD如何？Q/EONLY如何？
- 真实Path是否优于独立T_QT及自身f0？E内容是否必要？QT-SUP/QT-KD、KD-NATIVE分别如何？若Path不成立必须明确，不改变主方法。
- strict evidence-only的新增目标是否保留至Top10，而非只U变大？
- 两seed是否都独立训练Teacher和Student？若只seed13，不声称多seed稳定。
- 历史B13对照是否同协议实际重评？不能把它写成训练parent。
- 本轮哪些设定是预定工作点而非数据支持的最优值？哪些额外阶段没有运行？

### 15.2 自动结束条件

完整seed13后的repeat gate失败：锁定仅seed13，完成原test的一次既定评测，导出所有已定表和决定即结束；不得追加“下一步诊断”并自动运行。两个seed完成：导出开发比较、锁定协议、一次test、最终包后结束。不能再开第三seed、更多epoch、第二轮Teacher反馈或修改rho。

> **最后再次要求：执行本文件规定的每个阶段和约束。若某项无法执行，明确指出；不得让旧parent、旧train_fit、旧负例、旧分数以“缓存”“初始化”“兼容层”的名字重新进入训练。独立QT对照、旧式Student对照、两卡并行授权也必须按专章执行，不能漏做或相互改名替代。实现了函数不是跑过实验，跑过局部不是完成整链，分数高也不能替代fresh与真实Path要求。**


---

## 附录 A. 原数据字段接线与来源断言

这是字段适配合同，不是授权执行者自由发明标签。源码中可见的原数据模式如下；实际服务器字段若不同，必须在 `SCHEMA_MAP.json` 中逐项说明其原始等价字段与生成器依据，不能读取旧训练列表猜映射。

| 原artifact | 直接字段 | 本轮用途 |
|---|---|---|
| query_tables | table_id, split, source_table_id, columns, rows | query全集、官方划分、原可见内容 |
| data_lake_tables | table_id, columns/rows 或 source_table_ref | 全湖target；不是仅GT目标 |
| source_tables | source_table_id, columns, rows | 展开引用对象；不能补入query未提供的内容 |
| bridge_assets | asset_id, asset_type, content（text）, local_path/relative_path（image） | 全湖evidence对象内容 |
| qrels | query_table_id, target_table_id, rel, split, reason | G及GT类别，相关性值>0为positive |
| evidence_recoveries | query_table_id, target_table_id, query_row_id, split, evidence.asset_id | 原始query-specific W与来源位置 |
| recovery元数据 | recovered_attribute.hidden_in_query 等原生成器记录 | 仅用于明确的训练支持标注与评测，不作模型输入 |

1. 先用query_tables的**显式split字段**筛原train query IDs；字段缺失不能默认train。随后stream qrels/recoveries，仅保留这些q，检查记录split若存在必须一致。不要调用旧`calibration_query_buckets`。
2. G只接原rel>0。qrels/recoveries指向缺对象、W指向t∉G必须报冲突，不能union恢复目标进去制造新GT。
3. `reason == model_recoverable_join_column` 是上传生成器明确写出的implicit来源。其他reason不能仅凭“不等于该字符串”一概宣称direct：需要数据集本身的direct/explicit标注或原生成规则。无明确direct事实的(q,t)不进入D；不由QT分数补D。
4. query分类优先使用原dataset明确query_kind且核验与记录一致；仅有qrels reason时，全部正qrels均为上述implicit reason→implicit；具有明确direct事实且没有implicit正对→explicit；两者都有→mixed；仍无法判定→unknown。mixed/unknown保留overall并单列，禁止为凑599/599删掉或硬分；主要数据若无这些例外照实为0。
5. support margin仅在原GT明确hidden/implicit依赖的anchor上生效；D未知不是“已证明非direct”，不得仅凭D为空就给所有q加support loss。禁止把reason缺失自动补成implicit。
6. source_table_ref展开后保留被引用对象的原table_id与对象自身覆盖字段；若query引用整张source但自身有行/列选择器，必须严格应用选择器，不把整source直接当query。
7. text/image精确内容别名在规范化后将所有W、QE正集同步canonicalize；对象ID只供索引与join metadata，不能作为token字符串给模型。duplicate table不合并GT target IDs。
8. label builder把query所有原train监督处理完后，产出active task项；不能从只有witness的query反推整个train人口。新增训练图只是本run标签+检索派生产物。

`SCHEMA_MAP.json` 必须在正式训练前具有实际字段值与原始示例定位，不得保留TODO。映射是数据格式的确定性适配，不是可调实验因素。

## 附录 B. 冷启动、冷缓存与论文可复现性验收

**主命令的必要输入不允许包含“某轮实验先跑成功得到的文件”。**允许本run内部产生中间阶段，并允许纯内容缓存提速；这是两件不同的事。

- `cold_inputs`：不指定任何pure-cache路径，使用小型原始数据fixture及公共底座接口，必须能产生本版z、table行摘要、evidence摘要、标签、PCA、候选、随机Teacher和Student初始化。若本机无法加载公共底座，真实cold encoder测试标未执行，不拿mock成功说底座通过。
- `cache_inputs`：指定兼容纯缓存后，应复原相同对象输入定义和候选数据，不能因为缓存覆盖决定训练人口。对可复算的少量真实对象做数值容差检查，不要求再跑全湖底座。
- `no_history`：受限loader拒绝任何run外的task_checkpoint/optimizer/PCA/teacher_logits/training_list/learned_pooler_cache。公共backbone和经验证pure_cache例外。拒绝理由依据来源类型和父依赖，不仅是文件名包含B13。
- `new_run_dag`：PCA、原始训练列表、hard刷新和全部stage parent的leaf只能是原dataset/public backbone；同run缓存沿显式父依赖追溯；发现循环、外部任务父节点、无来源learned tensor即失败。
- `test_once`：两seed是否执行、所有终点与超参数已固定后，才允许final_test entry读取原test qrels。原test被历史开发看过的事实只披露，不重新随机划一个“新test”洗掉历史。

完整fresh的证据是这些输入/依赖合同及实际整链回执，不是把文件夹名字写成fresh，也不是优化器重新初始化。


## 附录 C. 历史最强工作点的比较方式与不能承诺的等价性

用户希望fresh QT路线接近先前最好checkpoint。保留以下四行，不能只列新Path而无旧式工作点：

1. 历史B13/已提供最强Student + 历史T0，在其原有完整可解析协议下的只读参考；缺资产注明不可执行。
2. 本轮 NATIVE-C1 + T_QT（中间参考）。
3. 本轮 KD-NATIVE + T_QT（正式旧式结构fresh对照）。
4. 本轮 KD-QE + T_QT 与 +T_PATH（固定检索器拆重排增益）。

所有本轮行用本轮原train、初值、特征、候选和完整GT分母。若历史表示/candidate/retention可精确适配本轮合同，再增加同协议重评；否则历史原协议只单列参考，不能把不同cache和准入下的差值当单因素效果。独立QT-SUP/QT-KD是无E输入对照，**不是历史B13的重新命名**，因为B13本来使用五关系和evidence检索。

`LEGACY_COMPARABILITY.json`逐项记录：dataset/split、z/局部token合同、PCA、Teacher结构和训练来源、Student结构和更新范围、loss/温度、列表构造、预算、ANN、retention/admission、评测人口是否一致。已见源代码 `mmdd_stage1/b13_recipe.py` 的旧SUP为10×sigmoid、KD权重0.3、P/R学习率分组等，并非本版默认raw logit/系数1；本版**不偷偷恢复它们，也不声称精确复现**。保留它作为说明差异的源码证据，不读取其历史路径或训练资产。

用户只授权在fresh前提下新增这些明确对照；“尽量接近旧最强”不是无限调参、接回旧权重或改主方法的许可。实际没有接近时如实报告；不能用增加E数据量或不同Teacher分支暗中补分后仍称QT-only。

## 附录 D. v2.1新增验收（与原P0/P1合并，不另起多轮诊断）

- T_QT/T_PATH起点tensor相同、optimizer不共享、两者均未加载历史任务权重；T_QT target loss未调用QET、标签用G不是D。
- QT-SUP/QT-KD仅表P/R，训练输入loader拒绝W/evidence，C1两臂完全同候选/顺序，C2完整目标双侧P梯度正确。
- KD-NATIVE没有adapter；固定e/t改变q时ET不变；更新P/R后key/index正确重建，不能从KD-EONLY改名。
- T_PATH-f0与独立T_QT分数缓存的key、checkpoint、表位不同；同池比较candidate人口一致。
- 严格Direct-only pipeline断言QE/ET调用次数为0、C100=D100；同模型Direct推理消融和独立训练Direct模型分开命名。
- 同卡/双卡独立作业不会共享RNG、optimizer、半写缓存或输出文件；profile不能改变正式初始化/样本顺序。
- 13节点DAG和最大26stage预算一致，主线的全部训练项与v2.0一致，新控制组不消费其数据RNG。

**以上是一次必要的实现验收，不新增方法搜索。执行每一项，不允许仅更新报告表头而不真正训练对照。**
