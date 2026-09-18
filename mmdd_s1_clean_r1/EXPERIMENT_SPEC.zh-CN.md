# MMDD Stage1 CLEAN-R1：从原始带 GT 数据集开始的完整训练协议

版本：1.0 · 日期：2026-09-17 · 状态：**新实验执行合同，不是已经跑出的结果**

本文优先级高于历史实验脚本、历史默认参数、旧 checkpoint 的 config 和上一轮讨论中的概念草图。源代码快照为 `mmdd_src_20260917.tar.gz`。当前对话没有上传完整原始数据湖或 Qwen 权重；本文给出明确的本地接线、实现、执行和验收合同，不声称已在真实数据上运行。附带 `reference_core.py` 是模型与数学语义参考，不是完整训练入口。

## 0. 这轮做什么、不做什么

**唯一主问题：一个单向、从零开始、基于紧凑对象缓存的 Teacher→Student 流程，能否在真实全湖检索中获得有效的 Stage1 收益，并使 evidence 参与最终目标判断？**

固定一个主训练湖：现有 **EntiTables 20K** 原始标注数据集。保留原始 train/dev/test 划分与 GT，不重新生成数据集、不重新让大模型标注、不混入 2K/200K 监督。与旧方案建议的 2K+20K 联训不同，这是本轮为收敛实验范围作出的明确选择。

实际训练只有三条轨迹：

| ID | 模型类型 | 初始化 | 作用 |
|---|---|---|---|
| T | 一个共享 Teacher | 新随机初始化 | 训练潜在目标/证据检索与当前证据支持判断 |
| S-SUP | 一个 Student | 新随机初始化 | 不蒸馏的必要对照 |
| S-KD | 与 S-SUP 相同的 Student | 与 S-SUP 完全相同的初始 state_dict | 主方法：SUP + 冻结 T 蒸馏 |

它们只有 **Teacher、Student 两种架构**，不是三种任务模型。T 只有一个 Transformer、一套输入投影、一个标量输出头。S-SUP/S-KD 是同一种模型的实验对照，部署只保留 S-KD 与 T。

本轮单 seed=13，不补 seed29，不做结构/温度/预算网格搜索；不训练 Stage2、column selector、生成器、证据 selector、resampler；不做 200K 的实际全湖前向或训练，只输出按实际对象数量计算的资源预算。不得把 query bootstrap 区间解释成多训练 seed 的稳定性。

**禁止读取**历史训练权重、优化器、PCA、projection、row-support 模型、Teacher logits、hard-negative list、retrieval rankings、历史对象选择清单或模型选择结果。禁止以 B13、C3、R22、R26、R30、T0 等为父节点。

**可以读取**原始数据、原始 GT、冻结 Qwen 官方预训练权重、满足第 4 节精确校验的纯 Qwen 对象特征缓存。旧缓存所在目录名称不重要，重要的是其内容是否纯属冻结编码而不是训练产物；缓存只能作为值查找，不能决定本轮对象集合。

## 1. 首先固定语义：潜在可发现性与当前支持，不强行用一个标签代替

### 1.1 两个问题，但不是两套网络

对 query Q、candidate x 和 evidence 集合 B，定义：

- `P(Q,x)`：x 是否值得作为目标/线索继续检索。x 为 T 时由目标 qrels 监督；x 为 E 时由原始 recovery 中出现的 Q–E 监督。
- `J(Q,T;B)`：**输入的 Q、T、B 中是否存在支持标注连接关系的可用线索**。它是 Stage1 的支持估计，不是完成行级 join 的形式化证明。

T 用任务 token `P` 或 `J` 调用同一个函数：

\[
 s_T(m,Q,x;B)=w^\top \operatorname{LN}(F_\theta(X(m,Q,x,B))_{\mathrm{REL}})+b,
 \quad m\in\{P,J\}. \tag{1}
\]

m=P 时 B 必须为空；m=J 时 x 必须为 table。m 不由 implicit/explicit 标签选择：**所有 query 在线执行完全相同的 P 检索/J 重排流程**。

### 1.2 处理“去掉 E 应不应该低分”

若 GT 声明 Q–T 需要补隐藏属性，则：

\[
 y_P(Q,T)=1,\qquad y_J(Q,T;\varnothing)=0. \tag{2}
\]

若有合法 direct GT，则：

\[
 y_P(Q,T)=1,\qquad y_J(Q,T;\varnothing)=1. \tag{3}
\]

因此，**值得召回不等于当前输入已支持连接**。本轮不会训练 P 把 implicit 正目标压下去，也不会训练 J 把缺少必要 evidence 的 implicit 空上下文当成已获支持。

对非空 B：找到原始标注 witness 时可以标支持；没有找到已标 witness 时通常是 `unknown`，不是负例。原因是 recovery 标注可能不穷尽全部有效证据。这不是说数据集没有 GT，而是目标 GT 与任意证据组合的 GT 粒度不同。

同一 Q–T 同时具有合法 direct 和 implicit 标注时，direct 优先决定空上下文支持=1；不能因其中一个 bridge 需要 E 就否定其另一个合法 direct 路径。

## 2. 原始数据接线与禁止信息

### 2.1 默认数据根目录

源码中可核对的原始 20K 根目录为：

```
<repo>/output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9
```

使用该目录中的原始产物，而不是 `work/stage1_*/lists`。路径搬迁时只改 `paths.dataset_root`；不得用名称相近的历史 list 顶替。Qwen 权重目录通过 `MMDD_QWEN_MODEL_DIR` 指定。若环境变量缺失，依次检查 `<repo>/models/Qwen3-VL-Embedding-8B` 与 `<repo>/Qwen3-VL-Embedding-8B`，首个含 `config.json`、完整权重索引及官方 wrapper 的目录可用；都不存在就明确报 `missing_backbone_path`，不得自动下载第二份权重。

原始产物优先通过 `mmdd_dataset.wdc_runtime.iter_dataset_artifact` 读取 dataset manifest 指定的分片；无 manifest 时读同名 JSONL：

| 产物 | 必需字段/作用 |
|---|---|
| `query_tables` | table_id、split、columns、rows、source_table_id |
| `data_lake_tables` | table_id、columns/rows 或 source_table_ref |
| `bridge_assets` | asset_id、asset_type、content 或本地图像路径 |
| `qrels` | query_table_id、target_table_id、rel、reason、split |
| `evidence_recoveries` | query_table_id、target_table_id、evidence.asset_id；可含 row/value/auto_check |
| 被 source_table_ref 引用的原始 source tables | 只用于恢复目标表内容 |

`evidence_recoveries` 无记录不会抹掉目标 qrels；但全湖都没有可用 recovery 时，本次预定的桥接训练不能成立，输入审计阶段停止，不改成 QT-only 后宣称完成。

### 2.2 建立当前输入清单

1. 对上述所有原始文件计算 SHA256；有分片时逐文件记录，并对有序清单再哈希。
2. 从原始 query、lake tables、assets 枚举完整对象集合；不得从旧 feature manifest、旧 target_lists、旧 fixed cohort 推导集合。
3. 解析所有 target 引用；保留原始目标 table_id。缺失被引用源表是硬错误，不得将其当成空列负表或删除后继续。
4. ID 必须全局唯一；query ID 不进入目标/evidence corpus。重名、GT 引用不存在对象、split 冲突均停止。
5. 只向模型序列化列名与可见 cell.text，以及 evidence 的原始文本/图像。`source_table_id/chain_id/target_table_ids/join_attribute/hidden_attributes/recovered_attribute/model_evidence/query_entity/source_row_id` 等标注元数据不得进入模型输入。可见表中真实 URL cell.text 是内容，保留，不统一替换成 `[url]`。
6. 每个 split 的 population 从原始 qrels 中 `rel>0` 的 query 建立；无正例 query 记录为 `no_positive_qrel` 并在训练前固定排除，不根据召回情况改变分母。
7. 从 qrels.reason 精确映射：`explicit_visible_join_column`→direct；`model_recoverable_join_column`→implicit。其他 reason 必须记录并报错，不能把未知 reason 自动当 explicit。
8. 依据原始 `source_table_id` 检查 train/dev/test query source groups 互斥。跨 split 的原始目标/evidence作为共同可见语料不等于标签泄漏；不得读取 dev/test 的 qrels/recovery 来生成 train 标签或过滤 train 负例。
9. 特征缓存可以覆盖所有 split 的无标签对象。训练 GT loader 只能打开已导出的 train supervision 文件；dev scorer 使用 dev；test GT 在模型、预算冻结后只交给 evaluator。

### 2.3 本轮新建的基础标签表

\[
G_Q=\{T:\operatorname{qrel}(Q,T)>0\},\quad
D_Q=\{T\in G_Q:\text{存在 direct GT}\},\quad
I_Q=G_Q\setminus D_Q. \tag{4}
\]

令 W(Q,T) 为原始 recovery 对应的 evidence **内容等价类**集合。只接受同时满足 Q/T/E 存在、Q–T 为该 split 正 qrel、record.split 与 query.split 一致的记录。

- 原始最终 `evidence_recoveries` 未带 auto_check 的记录，按数据集交付的 GT 使用，provenance=`dataset_gt_without_extra_review`；不能因为本协议偏好更强审核就把已有 GT 全部丢掉。
- 带 auto_check 时，reviews 非空且 verdict 全为 supported 才加入 W；缺失/冲突/拒绝的记录不作为 witness 正例，并分别计数。保留原审核策略名称，不把自动标注改称人工金标。
- recovery 中的列、值、行号可进入诊断旁路，但只把 `(Q,T,E)` 关系送入 Stage1 标签；不得作为模型输入。
- 不回退到“同 target 的其他 query 的 evidence”，也不回退到“同源图片/同源文字”。

定义：

\[
W_Q=\bigcup_{T\in G_Q}W(Q,T),\qquad
G_{Q,E}^{J}=D_Q\cup\{T\in I_Q:E\in W(Q,T)\}. \tag{5}
\]

对包含空集合在内的上下文 B，支持标签统一定义为：

\[
y_J(Q,T;B)=
\begin{cases}
1,&T\in D_Q,\\
1,&T\in I_Q,\ W(Q,T)\cap B\ne\varnothing,\\
0,&T\in I_Q,\ B=\varnothing,\\
0,&\text{存在该 Q–T 的明确非连接 GT},\\
\text{unknown},&\text{其他情形}.
\end{cases}\tag{6}
\]

这里“存在 witness”仅用已有标注监督逻辑桥接，不把一阶段 joinability 重新定义成 row-coverage 阈值，也不保证完整表每行均可执行连接。

## 3. 内容去重与确定性规则

所有字符串排序用 UTF-8 字节序。所有“随机抽样”由 SHA256 排序实现：`key=SHA256(namespace+'\0'+id)`，先按 key、再按原 id 排序。namespace 包含固定 seed、phase、packet、epoch、query_id、必要时 evidence_id，精确格式见第3.1节。禁止使用 Python 进程随机化的内置 hash。

文本 evidence 内容 key：原始 content 做 Unicode NFC、CRLF→LF、首尾去空白后 UTF-8 SHA256，不折叠内部空格、不改写词、不剥除实体信息。图像 key：文件原始字节 SHA256；不声称这能识别所有视觉近重复。

同内容 evidence 只在 corpus 保留最小 asset_id 作为 canonical ID；所有原始 IDs 映射到它，W 同步映射。text 与 image 不跨模态合并。候选 target 仅按 table_id 去重，不依据内容相似度删除 GT 目标。

空文本（上述规范化后长度=0）、无法解码图像不进入可检索 evidence corpus，写 failure registry。引用它们的 query/target GT 不从评测分母消失。非空短文本不按长度阈值过滤，本轮不新增数据筛选参数。

所有相同 score 的候选按 canonical ID 排序。所有排名先去重再取 Top-k。NaN/Inf score 为实现错误，不得静默换成 0。


### 3.1 命名空间、随机性与排序精度

随机列表的namespace固定为 `clean-r1:13:<phase>:<sampling_arm>:packet=<D/E/C/B>:epoch=<e>:q=<Q>:anchor=<E或none>:<purpose>`。phase取teacher/student；Teacher sampling_arm取T，两个Student臂的sampling_arm均取S（不取SUP/KD）；purpose取uniform或list-order。因此在相同HardRank和同一合法剩余集合下，SUP/KD抽到相同随机竞争项；后续因各自挖掘的HardRank变化而出现的列表差异是预定的流程效应，不额外引入arm专属随机数。每epoch query遍历顺序使用 `stable_order(query_ids, namespace="query-order:13:<T或S>:epoch=<e>")`，两个Student臂使用同一S顺序。HardRank按真实score降序/ID升序，不另加随机扰动。witness-cycle与B-view使用第7节给定的特殊namespace，不额外包含arm，使SUP/KD选择同一epoch的witness anchor。SHA256排序在全部合法剩余corpus上执行，因此是确定的无放回抽样，不是只在“容易负例池”内抽样。

每条轨迹开始同时设置Python random、NumPy、torch CPU/CUDA seed=13；dataloader workers=0。初始化Student时重新设置seed，不继承Teacher用过的RNG进度；先生成一个student_init.pt再分别加载到两个新对象中，不共享可变参数存储。各轨迹开始训练时RNG再置13。恢复则读取同run的RNG状态，不再次置seed。

GPU float32 exact mining/metric重算禁用TF32；记录PyTorch/CUDA/驱动版本。BF16训练不能被描述为跨硬件位级一致。候选列表按float32真实分数排序，最终J指标用保存的float32 logits；统计累积用float64。tie只指该保存精度下严格相等，不以人为epsilon扩大tie。

## 4. 冻结预计算：只保存 8 个摘要，不保存完整 hidden states

### 4.1 明确缓存对象与尺寸

每个对象只持久化：

\[
z_x\in\mathbb R^{4096}\ (float32),\quad
C_x\in\mathbb R^{8\times4096}\ (float16),\quad
mask_x\in\{0,1\}^8. \tag{7}
\]

输入小模型时临时拼接 `V_x=[z_x; C_x]`，共 9 个 slots；z 不再复制一份落盘。附带 8 个 kind IDs 和内容/编码配置哈希。

不保存 KV cache、所有层 hidden states、全长 last_hidden_state、独立每行 Qwen embedding、Q×T、Q×E、Q×E×T 特征。**不得把一个训练中的 pooling/resampler 的输出当作长期冻结缓存。**8 摘要由固定非学习规则产生。

### 4.2 固定 Qwen 输入约定

使用现有 `cache_stage1_features.py` 的 `EMBEDDING_INSTRUCTIONS` 原文及 role 对应规则，版本 `role_modality_v2_object_only`，复制到新模块并记录 SHA256，不能 import 一个会间接载入历史实验状态的 run 脚本。

Qwen 冻结、eval、inference_mode，BF16 前向，use_cache=False、output_attentions=False；直接获取最后层输出，不请求所有层列表。最终 z 使用官方 last-valid-token pooling 后 float32 L2 normalize。模型 revision/权重哈希、wrapper 哈希、processor/tokenizer 哈希都写入缓存指纹。官方模型卡确认 8B 的原始向量维度为 4096；具体 API 以本地锁定 wrapper 为准。

固定 preprocessing：max_length=8192；图像 max_pixels=524288，保留官方 aspect-ratio resize 逻辑，不新做 crop/OCR/caption。必须在 processor 的真实处理参数中设置并记录 resize 后 grid，而不只是给 Python 对象挂一个没被读取的属性。单张图像，不使用视频。

表对象只序列化实际可见内容，调用已有 `serialize_table_parts` 的逻辑：max_rows=12、max_cell_chars=1024、row_format=`values`。query 原有不足12行时全部保留；超过12行时取原始顺序前12行并记录 truncation。schema/header 与每行分别作为一个 part。每个 part 再按本地 tokenizer 限制为前512个 token，decode为文本后拼接，避免一行长文本挤掉后续所有 example rows；记录每个 part 的原长/保留长。文本 evidence 限制为正文前7168个token。模型包装后超过8192应在 CPU 预检查时失败，不进入“反复前向再缩短”循环。

这些是本轮新固定的输入容量，不是声称历史输入已经一样。若与旧冻结缓存的实际输入不相同，就需要一次性新前向，不能复用数值。

### 4.3 摘要公式与 token 选择

只对内容 token pooling，排除 system instruction、chat markers、padding、assistant generation prompt。

**Table**：用 tokenizer offsets 精确定位 schema 与每行的 token spans；可复用 `_table_token_groups` 的内容定位思想，但不能使用其多次 Qwen 重试分支。先算各 part 的 float32 mean：

\[
c_{schema}=\frac1{|H_{schema}|}\sum_{h\in H_{schema}}h,
\quad r_i=\frac1{|H_{row_i}|}\sum_{h\in H_{row_i}}h.\tag{8}
\]

C[0]=schema；其余7槽按原始行序号分组。m≤7时每行一槽，余槽padding。m>7时第j组为行序列切片 `[floor(jm/7):floor((j+1)m/7)]`，j=0…6，各组对 r_i 等权平均。所有非空摘要最后转float16。

**Text**：正文内容token序列长度L，分成最多8个连续非空区间。L≥8时第j区间 `[floor(jL/8):floor((j+1)L/8)]`；L<8时前L槽各一个token，其余padding；逐区间float32 mean后转float16。

**Image**：只取语言模型最后层中对应 image placeholder/merged visual token 的输出，不取所有 prompt token，也不取另一个维度的 vision tower 原始输出；按 processor 产生的 visual-token 顺序，使用与Text相同的8区间pooling。若对应token计数无法与processor grid/merge规则核对，报 `image_token_mapping_mismatch`，不得悄悄退成对全部prompt求均值。

kind IDs：0=GLOBAL(z)，1=SCHEMA，2=ROW_GROUP，3…10=TEXT_OR_IMAGE_BIN_0…7。表行组不添加绝对行序位置embedding；text/image的有序bin类型通过kind保留。

序列化后 token offsets/visual positions 先在内存中验证。完成 C、z 后立刻释放全长 H，不写临时 H 文件。缓存构建只给一个对象版本执行一次成功的 Qwen 前向；OOM 允许把后续batch二分重试，但不能改变输入、token预算或产生“短输入替代长输入”的隐式结果。

### 4.4 复用与硬性资源预算

允许从旧**纯冻结**缓存计算8摘要，前提是能核对模型/processor/prompt/精确输入指纹，并且旧缓存含足够的原始 token 信息或相同结构 part 均值。只有旧 global z、旧learned-resampler输出、或者H长度却无可核对token位置，都不足以重建本协议 C。存在不兼容时只对缺失对象前向一次；禁止训练每epoch补跑Qwen。

额外特征字节（不含metadata）为：

\[
B_{feature}(N)=N\,[4096\cdot4+8\cdot4096\cdot2]=81,920N.\tag{9}
\]

| 对象数N | z+C |
|---:|---:|
| 20,000 | 1.526 GiB |
| 200,000 | 15.259 GiB |
| 600,000 | 45.776 GiB |

**N是表+文本+图像+query去重后的总对象数，不是只数表。**因此不能把15.259GiB当成任意200K表数据湖的总缓存。metadata、ANN图、原始图片、checkpoint、训练输出另计。

每4096对象一个shard，分别保存z.f32/C.f16/mask.u8，memory-map读取，不为每个对象创建大量小pt文件。CPU对象LRU最多1024对象，不把整个H池载入内存。

本轮新增 feature 区上限60GiB；整个新增experiment workspace上限100GiB；始终保留至少200GiB文件系统剩余空间。预计算之前按实际N及metadata裕量2%计算；超预算停止并输出数目/字节来源，不自动删用户文件、不换4槽、不丢模态、不缩湖。每写完一个shard再次检查。只保留一个shared frozen cache，不为T/S-SUP/S-KD复制缓存。

## 5. Student：一个对象池化器，一个受Q条件影响的ANN查询

### 5.1 对象表示

维度d=1024；输入D=4096；三个模态共享W_S（不为每个任务训练独立encoder）。定义无参数RMS归一化：

\[
N(x)=x/\sqrt{\operatorname{mean}(x^2)+10^{-6}}.\tag{10}
\]

\[
h_{x,k}=\operatorname{LN}(W_S N(V_{x,k})+e_{mod(x)}+e_{kind(x,k)}),\tag{11}
\]

\[
\alpha_{x,k}=\operatorname{softmax}_{k:mask=1}
(a^\top h_{x,k}/\sqrt d),\qquad
\nu_x=\operatorname{unit}(\sum_k\alpha_{x,k}h_{x,k}).\tag{12}
\]

unit(x)=x/max(||x||₂,1e-6)。valid global slot必须存在。这个a只是同一Student中的一个1024维pooling参数，不是另一个网络/另一次训练。没有额外Transformer、MLP或resampler。

目标表索引key=ν_T，evidence索引key=ν_E，每对象一个向量。Table/Text/Image索引可以物理分开；不建行、列全湖索引。

### 5.2 三种查询，仅第三种联合Q/E

\[
u_D(Q)=\operatorname{unit}(A_D\nu_Q),\quad
u_E(Q)=\operatorname{unit}(A_E\nu_Q).\tag{13}
\]

低秩条件继续检索：r=64，

\[
u_C(Q,E)=\operatorname{unit}
\left(B\nu_E+U[\tanh(V\nu_Q)\odot\tanh(W\nu_E)]\right).\tag{14}
\]

A_D、A_E、B∈R^{1024×1024}；V、W∈R^{64×1024}；U∈R^{1024×64}。⊙是逐元素乘积；tanh逐元素。这里U/V/W是一个明确的低秩三元交互项的因子，不是三个独立模型。

\[
s_D(Q,T)=u_D(Q)^\top\nu_T/0.07,
\quad s_E(Q,E)=u_E(Q)^\top\nu_E/0.07,
\quad s_C(Q,E,T)=u_C(Q,E)^\top\nu_T/0.07.\tag{15}
\]

ANN用未除0.07的内积，排序相同；训练固定温度0.07，不学习scale、不加query-independent target bias。最终所有索引key与查询向量L2归一化；对任何有效对象/查询的unit前范数<1e-6（包括全零）或非finite值报数值错误。

**E改变的是当前query向量，不改变目标key或目标索引。**这与对每个Q重建index不同。训练期间模型参数更新后按第9节刷新向量；部署期间模型冻结，索引静态。

这里不把s_D与s_C相加、不做跨路径LSE、不将QE分数乘入ET；第10节用预算调度合并候选，最终评分由T完成。

### 5.3 初始化

所有新增参数torch seed=13。W_S/U/V/W用Xavier uniform；A_D/A_E/B初始化为单位矩阵；模态/kind embedding正态std=0.02；a=0；LayerNorm weight=1,bias=0。不加载旧PCA或旧raw-aligned投影。S-SUP与S-KD从同一份本轮 `student_init.pt` 开始，优化器各自从零开始。

最终L2归一化约束检索分数范围，但不能防止条件残差主导向量方向，也不能证明不会发生方向hub。为此还必须刷新全湖负例并执行第12节hub诊断。

## 6. Teacher：一个共享三层Relation Transformer

维度512、heads=8、FFN=1024、layers=3、GELU、dropout=0.1、pre-norm、LayerNorm eps=1e-6、final LayerNorm。

对象slot输入：

\[
t_{x,k}=W_T N(V_{x,k})+e_{mod(x)}+e_{role(x)}+e_{kind(x,k)}.\tag{16}
\]

三个role：Q、CANDIDATE、E_CONTEXT。W_T∈R^{512×4096}对所有对象共享。

P调用：`[REL+task_P], Q的9槽, candidate的9槽`。candidate可以是table/text/image。

J调用：`[REL+task_J], Q的9槽, T的9槽, B中每个E的9槽`。

送入Teacher之前将B按canonical evidence ID排序（检索RR仍使用原检索stream顺序）。第一层仅允许**各对象内部**attention，[REL]仅看自身；第二、三层允许所有有效token跨对象attention。这样保留各对象的slot分组后，再联合交互。无全局绝对position embedding，无evidence ID embedding，无source ID embedding。对象顺序变化时应随之置换mask，不能改变role。

最长自然B=20时有效长度上限1+9+9+20×9=199。空B不插入一个会被误当证据的global z；使用真实空对象集合，batch padding完全mask掉。

输出统一按式(1)。P与J只有两个task embedding不同；不是独立Fbase/Fbridge、没有独立gate、没有单独bundle聚合网络、没有第二个scoring MLP。输入投影、Transformer、单线性输出头在同一optimizer内联合训练，保存一个checkpoint。

初始化：每层独立Xavier uniform（不能复制相同layer随机参数）；embedding与REL std=0.02；norm weight=1/bias=0；其他bias=0；readout Xavier uniform，bias=0。初始化顺序与 `reference_core.py` 对齐。

### 6.1 Transformer内部算子（固定实现语义）

每个pre-norm层，令输入H，d=512，8个head、每head维度64。采用row-vector记号，公式中的右乘矩阵对应PyTorch存储weight的转置。对每head：

\[
 Q_h=\operatorname{LN}_1(H)W_h^Q+b_h^Q,\quad
 K_h=\operatorname{LN}_1(H)W_h^K+b_h^K,\quad
 V_h=\operatorname{LN}_1(H)W_h^V+b_h^V,
\]
\[
 A_h=\operatorname{softmax}(Q_h K_h^{\mathsf T}/\sqrt{64}+M),
\]
\[
 H'=H+\operatorname{Dropout}\left(\operatorname{Concat}_h[\operatorname{Dropout}(A_h)V_h]W^O+b^O\right),
\]
\[
 H_{next}=H'+\operatorname{Dropout}\left(\operatorname{Dropout}(\operatorname{GELU}(\operatorname{LN}_2(H')W_1+b_1))W_2+b_2\right).
\]

所有dropout p=0.1，只在train启用；M允许位置为0、禁止位置为−∞。投影Q/K/V和FFN按PyTorch `TransformerEncoderLayer`默认含bias。第一层mask按照完整9slot对象块而非模态块划分，不允许不同evidence只因都是text就互相看见；后两层全局。

全padding的对象块仍需避免padding query出现“所有key都被mask”导致NaN；可以让这些无效query仅访问REL，但它们始终不能被有效query作为key读入，等价于删除padding。该例外不允许有效对象token在第一层访问REL。reference实现包含此处理。

与reference默认维度一致的参数量：Teacher **8,417,281**；Student **7,554,048**，均不含冻结8B backbone。它们是当前CPU实例化计数，不是预估训练效果或GPU显存峰值。原始浮点权重约32.11MiB/28.82MiB；optimizer、activation、输入batch另计。

## 7. 从原始GT新建训练组合：不是重新标注

### 7.1 初始无监督检索

用本轮对象全集的纯Qwen z，在float32下分块exact内积检索。query batch=64、corpus chunk=4096；不创建Q×全湖的长期矩阵。

- `D_raw(Q)`：Q→T前100；另外为训练保存前128目标IDs。
- `E_raw(Q)`：Q→text前10、Q→image前10；每模态额外前128用于负例候选。
- 每个进入E_raw的E，用z_E→T前20构造raw两跳结果；训练单E的candidate pool另保存z_E→T前128。
- 对训练选择的GT witness E不在自然E_raw中的情况，允许**离线训练**计算其z_E→T前128；不因此把它插入自然dev/test检索。

这些排名全部在本轮从原始对象重新计算；只有检索这一步不用标签，训练组装下一步才读取train GT。

### 7.2 统一列表采样操作

`MakeList(anchor, P, Excluded, HardRank, epoch)`：

1. P是该任务完整已知正例集合，按ID排序保留全部，绝不把其余正例充当负例。
2. 从HardRank中跳过P、Excluded、非法IDs，取前16个不同负候选。
3. 从合法destination corpus剩余对象中按稳定hash抽15个。
4. Hard不足16时由hash池补齐；hash池也不足时取所有剩余，不重复、不用padding假装负例。
5. 列表是 `all_positive_ids + 31 negative_ids`，所以总长度是 |P|+31，不是强制32。生成完按稳定hash重排整个列表，模型不可读取位置标签。
6. `Excluded`仅表示该任务应mask的其他已知正目标；不得用dev/test标签扩展它。

benchmark中未列为正qrel的目标，可作为**contrastive competitor**参与ranking loss，但记录 `label_provenance=sampled_unjudged`，不转换成已审核不可连接GT。明确rel≤0 qrel可记录`confirmed_negative`。证据同理。BCE只使用明确GT/原始必要证据定义产生的0/1，不使用sampled_unjudged的0标签。

这是一个正例+未判定对比学习假设，不是关于全湖负标签完备性的事实宣称。必须报告各种负候选占比。

### 7.3 每个query每epoch的四种packet

所有epoch按稳定hash排列train query，每个query恰好一次；每次依次构造以下packet。

**D packet**：P=G_Q；destination=所有T；HardRank为raw QT或规定的刷新排名；Teacher调用P(Q,T)。

**E packet**：P=W_Q；destination=所有可用canonical evidence（text和image合并）；hard pool将两模态raw前128按名次交替，不比较两模态未校准分数。P为空则该packet不执行并计数。Teacher调用P(Q,E)。

**C packet**：从排序后的W_Q中选一个E。先用namespace=`witness-cycle:13:Q`对W_Q稳定排序，epoch e（1-index）取位置(e−1) mod |W_Q|。P=G^J_{Q,E}；Excluded=G_Q\P；HardRank=该E的raw ET或规定刷新排名；Teacher调用J(Q,T;{E})。P为空则不执行。这里只每epoch采样一个witness anchor以控制成本；基础W表是完整的，必须导出六个epoch累计被采样的witness覆盖，不声称一轮遍历了所有三元组合。

**B packet**：使用与D packet**相同的target列表IDs**，而不是为正目标换一套evidence。

- natural B是原始Q→E每模态10条的20条去重结果。
- e=1,2时不执行B packet，先学习D/E/C与必要证据对照。
- e=3…6时，若 `(e + h_Q) mod 2 = 0`，使用natural B；否则使用witness-augmented B。`h_Q=int(SHA256('B-view:13:'+Q)前16hex,16)`。
- augmented B：使用本epoch C packet的E。E已在B时不变；否则替换同模态B中最后一条（按当前Q→E名次），该模态不足10条则追加。无其他同模态candidate时也可追加，但总数≤20。没有W_Q时使用natural B。
- **所有候选T共享同一个B**，不能只给正T补GT evidence；GT只影响离线训练view的构造，不成为模型特征。
- J正目标 P_B=D_Q ∪ {T∈I_Q:W(Q,T)∩B非空}；其他G_Q成员mask，不当负例；负候选仍为D列表中的非G_Q目标。
- 不管P_B有几个，都对所有P_B计算式(17)。若P_B为空，该B packet的ranking loss=0并记录。

以上natural/augmented view比例是固定训练暴露安排，不是在线的GT路由。dev/test禁止augmented view作为正式结果。

## 8. Teacher训练损失、刷新与选择

### 8.1 多正例ranking loss

对于正集合P、负竞争集合N和score s，

\[
L_{rank}(s;P,N)=\frac1{|P|}\sum_{p\in P}
\log\left(1+\sum_{n\in N}\exp(s_n-s_p)\right).\tag{17}
\]

P/N不相交。实现用logsumexp，常数1对应一个零logit。P或N为空时返回可微0，并写skip原因。不使用 `−log(sum probability of positives)`，避免只推高一个容易正例便满足所有监督；也不让多个已知正例互相竞争。

分别计算L_D/L_E/L_C/L_B。每个Q的base loss是本阶段实际存在且P/N非空的packet loss的算术平均；没有全部四项时不能将缺失项计作真实0稀释其他任务。先由标签mask确定有效packet数与cal有效类别数，再执行前向/反向。C packet与其复用的cal正项合成同一次backward，不能第一次backward释放图后又对同一logits执行第二次backward；不为此跨query保留完整计算图。

### 8.2 必要证据校准，不额外训练“支持模型”

每个Q有三类可用对照：

A：对D_Q所有T计算 `softplus(-J(Q,T;∅))`，目标1。

B：对I_Q所有T计算 `softplus(J(Q,T;∅))`，目标0。

C：对本epoch选中E对应的 `T∈I_Q且E∈W(Q,T)` 计算 `softplus(-J(Q,T;{E}))`，目标1。

每类先对该类T取均值，再对存在的类别取均值，得到L_cal。空类别不计入平均。

\[
L_T(Q)=\operatorname{mean}(L_D,L_E,L_C,L_B\text{中的有效项})+0.2L_{cal}.\tag{18}
\]

C与L_C使用同一批singleton logits，不能把它们detach或读旧缓存。空B的J logits是新的mode=J前向，不得用P(Q,T)顶替。此对照是训练“缺必要证据时低支持”，不是给potential P标负。

### 8.3 六个epoch的单向训练日程

T从随机初始化开始：epoch1–2 D/E/C+cal；epoch3–6 D/E/C/B+cal。全程一个optimizer，不重置epoch间动量，不换随机初始化、不引用历史父checkpoint。

epoch2结束后只做一次**本轮Teacher内生的负例刷新**：

- D hard pool固定为raw QT前128与按第10节RR得到的raw evidence-target排名前128目标的自然并集；在池内用当前T的P(Q,T)重排。
- E hard pool固定为每模态raw QE前128的并集；用T的P(Q,E)重排。
- C hard pool为对应E的raw ET前128 ∪ raw QT前128；用T的J(Q,T;{E})重排。为epoch3–6将选中的有限E分别准备，不枚举W_Q×全湖。
- 刷新时排除所有该task已知正例和masked positives。GT只属于train。
- 仅更新后续epoch使用的HardRank；random negatives仍按各epochhash重新抽样。
- 冻结natural B仍是raw QE20，不让Teacher挑过的GT witness伪装成自然供给。

这一步是预定的顺序阶段，不涉及Student，不形成T↔S循环。

优化器AdamW，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01（bias/norm/embedding/REL/pool_query不衰减）；global grad clip=1.0。每个optimizer step累积4个完整query loss，取算术平均；最后不足4个按实际数目平均。query内部多个packet可分别backward加权贡献，但不能retain_graph累积所有大前向。BF16 autocast，归一化、loss、softmax、累积loss为float32；FP32 master weights。前5%预定step线性warmup，之后cosine到0.1×base LR，全部step数在训练前由query数和6epoch确定。

dev候选固定为第10节的raw C100、raw B20；每epoch结束计算T的J完整重排。选best的字典序键为：overall R@10 → implicit R@10 → overall R@20 → 更早epoch。metric保留double，不根据四舍五入数值破同分。若dev没有implicit正例，该tie-break位置取−∞并在报告写N/A；不能改为读取test或临时换分项。T最终冻结best，继续保存last用于审计。不得根据test选择。

不因为dev分数不好就更换公式、增加epoch或改主模型；数值/输入错误停止，效果差则继续完成预定Student对照并如实报告。


### 8.4 学习率、分块与归约细节

对一个arm，Q数为n、effective query batch为b，step总数S=6 ceil(n/b)，warmup步数W=max(1,ceil(0.05S))。第t步（1-index）的倍率为：

\[
\eta_t/\eta_0=
\begin{cases}
 t/W,&t\le W,\\
 0.1+0.9[1+\cos(\pi(t-W)/(S-W))]/2,&t>W.
\end{cases}
\]

当S=W时只有warmup分支。第t次optimizer.step使用该t对应的lr，而不是step后才设本应使用的lr。epoch间不reset scheduler。Teacher和Student使用各自的n/b/S计数。

一个packet的候选可以按8个T/E分块进行Teacher前向，但在所有有效候选logits拼接后计算同一个list loss；不能把每8个候选各自softmax再平均，这会改变负例竞争。query内不同packet可以按事先算出的权重顺序backward；完整query loss的归约始终与式(18)一致。最后不足effective batch时，除以实际query数。

单卡A100顺序运行。构建cache时不加载T/S；训练T/S时卸载Qwen权重。Teacher fixed logits需要评分时只加载冻结Teacher与当前Student两套小模型，不加载8B。训练数据不触发动态Qwen前向。若实际显存不足，可减小仅用于tensor分块的teacher_target_chunk（8→4→2→1）并通过reference等价性检查；不改effective query batch、候选IDs、证据数、token数或loss分母。这个分块适配不是算法参数搜索。

## 9. Student训练：SUP和SUP+KD两条同架构新轨迹

### 9.1 输入、正负例与损失

D/E/C packet的P、Excluded、witness cycle与第7节一致；不训练Student的B packet，不给Q-only Student蒸馏看到20条E的Teacher bundle logits。

分别对三个packet用式(17)计算supervision；Student score使用式(15)。对应Teacher：

\[
D:\ P_T(Q,T),\quad E:\ P_T(Q,E),\quad
C:\ J_T(Q,T;\{E\}).\tag{19}
\]

同一个packet中T/S候选IDs、顺序、mask、Q与E输入完全相同。C中其他已知G_Q但未标该E支持的T既不在negative，也不进入KD分母。单次packet在valid=P∪N上的温度分布：

\[
p_T(i)=softmax(s_T(i)/2),\quad p_S(i)=softmax(s_S(i)/2),\tag{20}
\]

\[
L_{KD}=2^2\sum_{i\in valid}p_T(i)\log\frac{p_T(i)}{p_S(i)}.\tag{21}
\]

Teacher logits显式detach、Teacher eval，不能把教师分类标签硬改为one-hot后仍称KD。S-SUP优化有效packet的L_rank均值；S-KD优化有效packet的 `(L_rank+1.0*L_KD)` 均值。固定tau_KD=2、lambda=1，不扫参。

### 9.2 每epoch面对全湖竞争，不复刻固定32列表失稳

每条Student轨迹共6epoch，各自从同一本轮student_init开始。epoch1 hard negatives用本轮raw排名。epoch e>1开始前：

1. 用该轨迹epoch e−1的**last**参数，对全体目标与evidence的紧凑缓存重新计算Student keys；不跑Qwen。float32 keys落盘单个当前bank，记录model hash。
2. 在全湖进行exact top128挖掘：D按u_D(Q)，E按u_E(Q)，C按u_C(Q,E)。只对本epoch会使用的query和witness anchors挖掘。
3. 按MakeList取16 hard+15 uniform competitors，完整排除该task的已知正集合。
4. 本epochbank只用于选择IDs，不参与loss计算；训练前向用当前参数重新编码列表内全部Q/E/T。因此没有“给current query接一个未声明的旧key梯度”的错误。
5. 下一epoch重新生成bank和hard IDs。当前step内不对top-k/ANN反传。

T始终冻结。对同一 `(T_hash,mode,Q,E_or_empty,candidate)` 的single-object或singleton分数可以SQLite缓存；不保存全长H、不建立全湖Q×E×T数值表。逐epoch只追加真正使用的条目；超过workspace预算停止，不静默丢日志。

两条轨迹的epoch>1候选可能不同，因为各自模型不同；这是本轮要观察的**蒸馏完整流程效应**，不能包装成“固定训练列表下纯loss因果效应”。额外固定dev候选比较见第12节，不新增训练臂。

### 9.3 优化器与选择

AdamW lr=2e-4，其他优化器/衰减/clip/AMP规则同T。每step累积16个完整query loss；余数按实际query数平均。6epoch，5%warmup+cosine到0.1×baseLR。所有三类检索参数联合更新，不冻结历史base、不额外训练残差网络。

每epoch结束，用当前Student执行第10节完整自然检索，并由同一个冻结T做J重排。选best的顺序同T：dev overall R@10 → implicit R@10 → overall R@20 → 更早epoch。用于next-epoch mining的是last，不是best；最终部署的是预先规定选择的best。两者不得混淆。

训练过程中Teacher成绩不够好也不能静默将lambda调成0；S-SUP就是明确的无KD对照。

## 10. 在线检索、候选合并与最终重排：完整且不混合异义logit

### 10.1 索引

hnswlib，space=`ip`，dim=1024，M=32，ef_construction=200，ef_search=256，random_seed=13，按canonical ID排序插入，construction threads=1，query threads=1。加载index后显式重新set_ef(256)。保存index与key-bank/model/corpus hashes。

主结果用ANN真实流程；另做exact同模型诊断。HNSW返回distance=1−dot，以实际dot重算候选score排序。Top-k不能超过corpus大小。不自动调ef。raw基线同样建4096维ip索引并使用相同HNSW参数，另外报告exact raw结果以区分维度造成的ANN误差。

### 10.2 每个query执行的确定步骤

1. 获取/生成Q的9slot紧凑表示（在线新Q一次Qwen前向）；Student计算ν_Q。
2. `D100=ANN_T(u_D(Q),100)`。
3. `E_text=ANN_text(u_E(Q),10)`，`E_image=ANN_image(u_E(Q),10)`，内容已在索引前去重。
4. 将两模态排名按text第1、image第1、text第2、image第2…交替，得到B_Q，最多20。一个模态耗尽则继续另一个，但不把该模态配额转给另一模态超过10。本步骤不读取T/GT。
5. 对B_Q中的每个E计算u_C(Q,E)，在同一个目标index取Top20列表L_E。
6. `R_E=RR(L_E1,...,L_Em)`；RR定义见下。保留每个T的**所有**自然arrival evidence IDs与各stream名次，不再截断为Top4，也不按另一个scorer换路径。
7. `U=set(D100)∪set(R_E)`。
8. `C100=RR(D100,R_E,limit=100)`；不足100则返回实际并集，不补GT、不再次检索。D100/R_E交替每次各提供一张尚未返回的T。
9. 对C100中的每张T，用**同一**自然B_Q调用共享Teacher：
   \[
   r_T=J_T(Q,T;B_Q).\tag{22}
   \]
   不仅给arrival path里的E；所有自然召回E均可供T检查。这样direct召回T也可能使用已自然召回的有用E。
10. 按 `(-r_T,target_id)` 排序；输出完整C100排名与其前50，主指标读取前10。所有T都有J分数；不因没有arrival path、没有标注witness或得分低而删除候选。

RR伪代码：维护每stream游标；按stream顺序轮转；先跳过已emit目标，再emit该stream的下一张新目标；到limit停；全部耗尽也停。**不是每轮各消耗一个重复条目便让另一stream多占预算**。附带reference函数为规范实现。

最坏自然target union≤100+20×20=500；Teacher最多看100个target，每个上下文≤199tokens。没有按C100 target在全湖重检索E；不存在Q×T×E全湖笛卡尔积。

本轮最终分数不使用`QE+ET`、LSE、MAX path、weighted RRF、P与J插值、table×column。RR只是固定预算调度，不被包装成joinability概率或主方法创新。通过独立U/MatchedDirect对照区分admission与rerank，见第12节。

### 10.3 Stage2交付接口（只输出，不执行）

每query输出：query_id，C100有序target IDs，Top50，J logits，完整B_Q的canonical/original evidence IDs与原文/图像引用，arrival_evidence_by_target，D100、R_E、各L_E、索引与模型hash。

Stage2原有最大表—列乘积选择保留：

\[
(T^*,c^*)=\arg\max_{T,c} softmax_T(r_T)\,\rho_{T,c}.\tag{23}
\]

这里不对列求和，不改成先选表，不执行column模型。本轮也不强迫现有Stage2立即使用新的全query B_Q；同时输出arrival bundle与full B_Q使下一轮可独立接线核验。

## 11. 运行顺序与命令合同

新增入口为 `src/mmdd_stage1_clean/__main__.py`，**当前旧仓库没有这个入口，需要先按本文实现**。以下是实现后应执行的命令，不是已有脚本名字的伪装。

```
export PYTHONPATH="$PWD/src"
export MMDD_QWEN_MODEL_DIR="<本机已存在的Qwen权重目录>"
python -m mmdd_stage1_clean audit-input --config clean_r1.json
python -m mmdd_stage1_clean build-objects --config clean_r1.json
python -m mmdd_stage1_clean cache --config clean_r1.json
python -m mmdd_stage1_clean raw-retrieve --split train --config clean_r1.json
python -m mmdd_stage1_clean raw-retrieve --split dev --config clean_r1.json
python -m mmdd_stage1_clean build-supervision --config clean_r1.json
python -m mmdd_stage1_clean train-teacher --config clean_r1.json
python -m mmdd_stage1_clean freeze-teacher --config clean_r1.json
python -m mmdd_stage1_clean train-student --arm SUP --config clean_r1.json
python -m mmdd_stage1_clean train-student --arm KD --config clean_r1.json
python -m mmdd_stage1_clean freeze-selection --config clean_r1.json
python -m mmdd_stage1_clean evaluate --split test --config clean_r1.json
python -m mmdd_stage1_clean diagnose --config clean_r1.json
python -m mmdd_stage1_clean package --config clean_r1.json
```

`<本机已存在的Qwen权重目录>`只是不在当前上传包中的环境路径；允许执行者通过指定的本地查找规则自动填入，不是让执行者自己选择算法。data_root和model_dir实际resolve后写入`resolved_config.json`，所有后续命令只读resolved版本。

各命令只读取其前置节点已完成的当前run产物；失败时不跳步。`train-teacher`内部执行一次epoch2刷新；`train-student`内部执行每epoch刷新。不得让外部agent临时编排额外训练。

恢复中断：只能从**同一run、同一arm、同一config/输入hash**的last恢复weights/optimizer/scheduler/RNG/sampler进度。fresh-run的`parent_checkpoint=null`；`resume_checkpoint`只用于中断续跑。不能把另一run的last当作resume。

推荐模块只有 `data.py/cache.py/models.py/sampling.py/train.py/retrieve.py/evaluate.py/__main__.py`。不新增通用plugin注册系统、模型工厂框架、复杂pipeline引擎或每轮一份复制脚本。旧Stage2和旧run脚本均不修改。

## 12. 必须输出的评测，不额外训练新架构

### 12.1 主指标与分母

对split中预先固定的query集合Q，

\[
R_q@k=|G_q\cap rank_q[:k]|/|G_q|,\quad
R@k=|Q|^{-1}\sum_{q\in Q}R_q@k.\tag{24}
\]

空排名/失败query贡献0，不从分母删除。目标先去重后取k。输出R@10主端点、R@20、CR@50，以及Recall(C100)、RawUnionRecall(U)、direct100 recall。implicit与explicit均报告：先对各query的该类正目标计算Recall，再只在至少有该类GT的query上macro。mixed query可分别进入两类分项；overall独立按全部G_q计算，不能简单平均两个分项代替overall。

### 12.2 正式对照表

| 方法 | 候选 | 最终排序 |
|---|---|---|
| RAW-D | raw Qwen direct100 | raw inner product |
| RAW-2H | raw Qwen两跳 + 同一RR C100 | C100 admission顺序（只作为无重排基线） |
| RAW+T | raw C100、raw B20 | 本轮T的J |
| SUP+T | S-SUP自然C100/B20 | 同一本轮T的J |
| KD+T | S-KD自然C100/B20 | 同一本轮T的J，**主方法** |

raw第二跳使用z_E，不人工拼Q/E向量；它是冻结原始embedding的两跳基线，不是另一个训练模型。

这些方法都同时报告ANN和exact路径的指标。精确检索不训练，不能只对表现差的方法临时换exact再混表。

### 12.3 固定候选诊断

在S-KD最终dev C100上固定candidate IDs和B_Q，额外计算：

- `P-QT`：本轮同一T，mode=P，无E；
- `J-empty`：mode=J，无E；
- `J-natural`：正式主分数；
- `J-shuffle`：把整个B_Q换成同split另一query的B，按query_id循环位移一位，无GT参与选择；只作扰动诊断，不自动称错误证据；
- 对原始W(Q,T)与自然B有交集的implicit正对，报告 `J-natural − J-empty` 的均值、中位数、正差比例；并报告被移除witness后剩余B，不把“没有已标witness”自动当已验证无支持。

同池顺序和所有target logits必须导出。P/J是不同任务的分数，不比较它们的绝对数值，只比较各自排名；J-natural与J-empty是同一任务，可以比较差值。

对S-SUP/S-KD还在相同raw dev C100上用各自s_D评分，报告同池目标排序指标，帮助区分训练检索表示与own候选变化；不称它隔离了完整KD训练的全部因果效应。

### 12.4 Evidence admission和公平direct预算

对S-KD：保存所有自然U和exact direct ranking。令M_q为exact direct前|U_q|，比较U和M的原始coverage；不调用更多Teacher即完成此诊断。另对`Direct100+T`使用同一个B_Q和同一个J，比较C100+T与Direct100+T的等100目标重排预算；这两者都允许使用E内容，区别仅是target admission。

strict EO：T∈G_q、T∈U、T不在**同模型exact Direct100**、且至少有一个自然L_E包含T。分别统计进入C100、J Top10/50、且arrival E与W(Q,T)相交的数量。分别报告目标数和query-macro，不用二者互换。

### 12.5 全湖稳定性与成本

每epoch输出：三类loss与KD entropy、每类有效packet数、每类positive/negative数、known-positive误作负例计数（必须0）、witness累计采样比例、查询/目标向量范数、C unit前范数、各关系exact Top1占用率最高目标及占比、Top10 distinct目标数。SVD仅对1024×64的U做奇异值谱，诊断方向集中；不以“未坍缩”替代召回结果。

在预先按hash固定的最多128个dev query上，同时跑ANN/exact；无W的query也保留用于D/E，C诊断只对自然E anchors。每epoch不能重新挑有利子集。正式test对所有query执行ANN/exact。

NN fidelity：|ANN_k∩Exact_k|/k，对D的k=100、E每模态k=10、C的k=20分别报告；它不是GT Recall。当fidelity低于0.95时报告ANN误差，**不自动调ef**。R@10差值为0也不说明ANN一致。

成本报告包含对象数分模态、缓存实际字节、每对象一次Qwen编码累计GPU时间、T/S训练GPU时间、Teacher缓存命中率、index build时间/字节、在线query encoder与检索/重排分项p50/p95、GPU峰值、CPU RSS。在线延迟排除离线cache build但不能把新Q的Qwen编码悄悄排除；同时给“已缓存Q检索后端”时间和“新Q端到端”时间。无实际新Q计时则标not_measured。

## 13. 统计与结论规则

所有方法共享冻结test population。paired bootstrap按原source group有放回抽样10,000次，seed=13。每次按抽中group的**重复次数**重复纳入该group各query，再在query层macro；不先合并同query去掉重复权重，不按行数加权。

主比较预先固定：KD+T − RAW+T、KD+T − SUP+T，指标overall R@10，同时报告implicit/explicit差值。其余为探索性诊断，不按显著性选择主结论。single-seed bootstrap不能说明初始化稳健性。

不设“必须涨多少”的成功承诺。若KD失败，报告SUP/KD性能及Teacher在对应条件列表上的强弱；若J-natural弱于P-QT，明确说本轮统一支持重排尚未建立优势，不改成P当主方法却保留J方法名；若U提升但C100不提升，归入admission/预算损失，不误称模型没检索到；若C100充足但J排序差，归入rerank；若缺W，报告支持标注覆盖边界，不声称所有自然E无效。

## 14. 硬性验收测试

开始正式训练前必须通过：

1. 读文件allowlist测试：任何历史checkpoint/list/logit路径被传入新训练loader即报错。原始冻结cache走独立feature-onlyloader，不允许反序列化旧optimizer/model字段。
2. GT分离：修改test qrels/recovery不改变train对象内容、train标签、train候选或train语义hash（无标签corpus本身相同）。全局INPUT_INVENTORY中的test文件hash当然会变化，不能错误地要求全局inventory逐字相同；train语义hash只覆盖train监督、所用无标签corpus内容和固定配置。
3. 其他正例mask：C中同Q另一E支持的GT target不能落入负集合或KD分母。
4. 式(17)对每个正例都有下降其loss的梯度；正例之间不竞争；无正/负可微0。
5. KD方向、温度平方、mask、Teacher detach匹配reference。
6. Student三个query向量/key范数为1；式(14)score等于静态key内积；改Q一般可改变C查询，不允许用E-onlyproxy。
7. Teacher参数只有一个transformer与一个readout；P/J mode共享同一个权重对象与checkpoint。
8. 同一Teacher对evidence块的排列在eval时不改变score（float32 atol=1e-5）；padding槽内容被扰动不影响score；所有有效对象组被mask正确。
9. J label测试：implicit空B=0，direct空B=1，implicit+已知W=1，implicit+未知非空B=unknown；potential对全部G=1。
10. RR有重复streams、空streams、候选不足时与reference一致；不会少补预算；不改变GT分母。
11. 原始target引用得到列内容；不得额外注入“哪一列是GT”、被query隐藏的GT答案或source ID等标注字段。目标表原本可见的正常列名/cell.text照常保留，即使其字符串碰巧等于GT列名/值；不能为了此测试把合法表内容删掉。
12. 特征dry-run随机每模态至少16对象：z和C形状、finite、dtype、prompt exclusion、image token映射及重复cache命中验证；模拟无GPU训练循环不得调用Qwen。
13. packet中E dropout/替换后，监督必须依据实际B重建，不能仍复用原正标签。
14. 训练/重排分数不得复用另一个checkpoint或不同B的缓存；键必须包含完整context内容IDs与缓存hash。
15. 候选不变应检查ID集合、数量和唯一性；Recall@50相同不是候选相同的充分证明。

附带CPU reference tests只覆盖其中的数学、mask、模型形状和RR，不覆盖真实数据/官方Qwen/cache/ANN集成；后者必须在本地实现后执行并保存回执。

## 15. 产物与交付结构

```
work/s1_clean_r1_20260917/
  resolved_config.json
  INPUT_INVENTORY.json
  SOURCE_SHA256.json
  raw_gt/{train,dev,test}.{qrels,recoveries,population}.jsonl
  objects/{objects,corpus,content_aliases,failures}.jsonl
  cache/{manifest.jsonl,shards/*,CACHE_RECEIPT.json}
  raw_retrieval/{train,dev}/...
  teacher/{init.pt,best.pt,last.pt,selection.json,training.jsonl,mining_epoch2/*}
  student_init.pt
  student_SUP/{best.pt,last.pt,selection.json,training.jsonl,lists_epoch*/...}
  student_KD/{best.pt,last.pt,selection.json,training.jsonl,lists_epoch*/...}
  teacher_logits.sqlite
  FINAL_SELECTION_LOCK.json
  test/{RAW-D,RAW-2H,RAW+T,SUP+T,KD+T}/...
  diagnostics/{fixed_pool,strict_eo,ann_exact,resource,necessity}/...
  tests/{unit,integration,leakage,cache}/*
  REPORT.md
  COMMANDS.jsonl
  MANIFEST.sha256
```

所有命令回执包含UTC起止时间、cwd、argv、git commit、源文件hash、config/input/checkpoint hashes、exit code、实际rows/steps、hardware、异常。不得只交RESULTS.md。完整raw rankings/scores/per-query artifacts必须包含于结果包或明确的文件清单；不能把失败query遗漏。

Checkpoint只保留init/best/last与必要的epoch2 T刷新身份hash；如果epoch2不是best不必永久存其weights，只保留其采样排名与hash。Student每epochkey bank只保留当前版本和最终所选部署版本，其余可由小模型重算；不删除原始数据或共享冻结缓存。

## 16. 对六个疑虑的明确处理

1. **去E低分**：J低、P仍可高；一个共享Teacher两种任务token。无在线GT类型分流，最终固定J重排，不把缺证据的potential当已证明连接。
2. **磁盘/前向成本**：每对象8摘要+global，one-pass构建；60GiB新增feature预算；按对象总数计算，不保存全长H，不每轮跑8B。
3. **模型数量**：只有一种Teacher、一种Student。Teacher三层共享Transformer+单线性head；Student单对象pooling+明确低秩条件内积，没有Fbase/Fbridge/gate网络集合。
4. **表×列选择**：原始定义确实是全局argmax乘积，式(23)保留，本轮不做Stage2；对固定非负P，`max_{T,c}P(T)ρ(T,c)=max_T P(T)max_cρ(T,c)`只是同一优化的数学改写，不是另一个算法；上一轮引入求和口径不适用于原设计。
5. **昂贵动作/evidence**：上一轮`a=(Q,T,c,row)`漏写了上下文。正确的后续动作接口应为`Recover(Q,row,c; evidence_bundle(Q,T),T_schema)`；先选表列，再构造实体-属性query，再在该bundle内做文本span/图像区域定位，最后恢复。这个是原方案已有次序，不是离线预计算所有Q/T/c/row/E组合。本轮只做z/C对象预计算，并完整输出B与arrival provenance；不执行、也不冒充已实现任何Stage2昂贵调度。
6. **干净lake**：保留现成目标GT/recovery，只把所有训练组合、负例、候选、checkpoint从零生成。不存在新增人工标注前置条件，不用旧list，不把“没有GT”作为本轮假设。

## 17. 依据与不确定性

原方案依据：`方案(4).md`的“一阶段粗召回”“二阶段验证”“Teacher/Student训练方案”；其中Stage2公式/步骤、冻结4096维、原始GT分档与200K效率定位被保留。最新历史分析表明目标级Path监督有效但仍弱于QT，故本轮改成直接联合Q/E/T内容判断，而不是复刻旧QE+ET聚合。历史结果不是新架构效果保证。

源码核对依据：`construction.py`的原始artifact读取、引用解析、GT字段及应避免的target/source evidence回退；`cache_stage1_features.py`的官方last_hidden_state读取和内容token定位；`diagnose_final_rerank_witness.py`的原始recovery字段；`evaluate_stage1_b4_test.py`的explicit/implicit reason枚举（只参考字段，不读取它的历史list/固定population）；`query_conditioned_et.py`的旧自由残差结构（新代码不复用其训练权重或lists）。

官方API参考：Qwen官方模型卡 `https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B`；hnswlib官方仓库 `https://github.com/nmslib/hnswlib`。这些来源只用于4096维/接口/索引语义，不提供本实验的性能依据。预算、模型尺寸、loss权重与训练日程全部是本轮明确预注册的设计选择，尚未调优。
