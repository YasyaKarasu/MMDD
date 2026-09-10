# MMDD R19：Global Teacher 定向扩容、困难负例与列表长度对照

**版本：2026-09-11 v2。本文完整替代 v1 作为执行依据；是尚未运行的实验方案，不是已运行结果。**

本次是对原方案的增量修订，不推翻 C0/C1/C2。保留原父权重、Global-1024 函数保持迁移、五关系检查、两个固定 seeds、固定末步、三池评估及 Stage1-only 范围。将 C2 明确为静态困难负例挖掘，新增 C3 仅检验同来源下 TT list 长度；fusion 只加无训练诊断，不新增自适应 RRF 训练臂。无需同时阅读 v1 才能执行本文件。

## 0. 本轮目标与范围

继续以 **Stage1 query-macro target Recall@10** 为唯一主要优化端点；报告 Recall@20、CandidateRecall@50。不运行 Stage2，不以 value recovery、correct join、Stage2 gate 决定本轮实验是否继续。

R18 最值得保留的结构是 A1 Global Residual，而不是旧 compressed-token-only Teacher。下一轮把三个竞争解释分开检验：

- **容量解释**：全局 object representation 压到512维，可能限制自然候选间的区分能力；只将 global_dim 扩到1024，local Transformer 维度仍为512。
- **训练竞争者解释**：保持结构、正例及列表长度不变，用冻结 Student 自然召回池和冻结 A1 挖掘高分 TT competitors，检验负例身份/难度来源。
- **竞争者数量解释**：在完全相同的静态 hard reservoir 中，保持 C2 原有候选，给 TT 列表追加不重复的新候选至总长32，检验更充分的 listwise 竞争。这里32含正例，不是32个负例。

本轮最多四个主训练臂 C0/C1/C2/C3、两个预先固定的 continuation seeds。**不做 global1024 × natural-negatives 或 global1024 × long-list 的组合臂，不做全网1024，不做 global+relation-heads，不改 Student、线上候选生成预算、主排序公式或 loss。** C3 在 C2 同一负例来源上只增加候选数量，是用于拆分来源和数量效应的受控对照；不能只报 C3−C0。

执行依赖：P0 correctness通过后，C0/C1可直接执行；另一支先冻结训练侧hard reservoir再执行C2/C3，最后统一Stage1评估与轻量fusion诊断。不要因mining尚未完成而挂起不依赖它的C0/C1。C3 不以 C2 精度先显著为触发条件；C2 短列表无效而 C3 长列表有效本身就是有价值的结果。只允许真实数据、功能正确性或资源缺口阻塞相应阶段。

## 1. R18 的事实依据及不能越过的结论

在同一批1,198 query、1,000 source group 的 natural union 中，独立复算结果如下，单位为百分比：

| 模型 | R@10 | R@20 | CR@50 |
|---|---:|---:|---:|
| A0 原结构续训 | 8.5977 | 17.4736 | 35.7888 |
| A1 Global Residual | 20.4229 | 32.5125 | 49.3114 |
| A2 五个 relation heads | 11.0462 | 19.3934 | 39.0025 |
| B13 固定同池参考值 | 29.0067 | 约35.08 | 约44.91 |

A1−A0 的R@10差值 +11.8253pp，独立 source-group bootstrap 95% CI约[+9.35,+14.38]pp，W/L/T=186/39/973。A1−A0 在 implicit/explicit 分别 +7.4569/+16.1937pp。A2−A0 的R@10仅+2.4485pp，主要来自 explicit；不能认为 A2 已解决跨模态 edge 问题。

使用同一个A1，U−matched-direct-M：R@10 +0.6260pp，CI跨0；R@20 +3.7841pp；CR@50 +4.0623pp，后二者CI为正。121个U-only正目标中，A1保留25个到Top10、69个到Top50，A0是10/59。

以上验证了“添加正确的全局表示通路值得继续”，**尚未验证512维就是瓶颈，也未证明local tokens必不可少或global compression是唯一根因**。A1比A0增加7,869,952参数，存在额外容量/优化路径的竞争解释。

三个注意点：

1. R18所选A0来自epoch1（5,268 updates），A1/A2来自epoch2（10,536 updates）。各臂训练预算相同且使用同一选点规则，但所评估模型并非相同update。因此需补固定终点比较，不能把已有结果称作严格same-step对照。
2. R18 `train()` 在 `initialize_arm()` 后才设 `torch.manual_seed`；新增global权重的初始化未被配置中的seed完整约束。现有保存checkpoint和排名有效，但从该配置重启不保证复现新增分支。此次修复不意味着旧结果作废。
3. 源码把固定1,198 query集合称为B13 dev set；它已参与多轮开发，不能改称 untouched test。R19不把这批query标签加入训练；后续论文最终结论仍需独立测试。

## 2. 执行依赖、输入与不需要下载的内容

默认工程根目录 `/home/oycy/MMDD`。允许通过 `--root` 指定根目录，但不得凭文件名替换checkpoint。下面路径均相对ROOT。

### 2.1 固定父权重与必要输入

| 输入 | 路径 | SHA256 |
|---|---|---|
| R19唯一父权重 | `work/stage1_optimization_r18_20260910/A1_global_residual/checkpoint_000002.pt` | `cefdd6be1d16ab86840d71e8fe55629cccc288a019df9efc33f93e4a852e16b9` |
| 训练列表 | `work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl` | `5be5e3aee605397c80b4bb43867e57d02d5e65147b150dc1275375946e543158` |
| edge dev列表 | `work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.dev.jsonl` | `d686c35d435631149a62b4f228c7d82c7ed6f5f5d42230149a11524bbb712de6` |
| 冻结评测候选池 | `work/stage1_optimization_r16_20260910/candidate_pools.jsonl.gz` | `4186b5bdd436a14fa61b83c3c6127507c075fd6f16804c5cdb2d4a06e85a01d1` |
| frozen feature manifest | `work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b/manifest.jsonl` | `c32099430feca4dae5d2f8fbbae60f965e3b0353fdd62c62a7bc929ba24216e1` |
| objects | `work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl` | `75f2789bfc62483c85dcecb3213ac6bd3f11b3952e15bba626a3990297e71e00` |
| R18 runner起点 | `src/run_stage1_r18.py` | `e793d3e29723e71e9453b1557db2e178899979093e42ff334dabfe87e7ea8d7e` |

继承R18的FeatureStore及teacher_extra解析方式：R12 `taskC_training/teacher_extra`；R16 `teacher_extra_matched_gpu0/1` 与 `teacher_extra_edges_gpu0/1`。这些是冻结Qwen特征，不等于可跨checkpoint共享的learned compressed-token cache。

B13 checkpoint从 `src/run_stage1_r16.py::_paths` 和对应manifest解析：

```
work/stage1_optimization_r13_20260909/
  taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt
  taskD_witness_supervision/p_s_target_only/evaluation_step178/index
```

Student用于C2/C3共用的训练候选挖掘，不修改参数/索引。只有同时需要相应destination type的已有索引时才加载；不能假定上述目录已有QE/ET所需的全部索引。按现有B13检索入口与manifest确认。如缺少，允许从既有冻结object vectors构建**参数和向量完全相同**的只读索引副本，并单独记录离线成本；这不是Student重训，也不得改变主评测池。

外部论文仅供方法背景，不是运行依赖。**不需要下载新的Qwen模型、外部数据集或论文附件。** 若现有某输入缺失，报告具体路径与影响范围；缺少C2/C3共用的训练自然池不应取消C0/C1。禁止自动换父权重、借用dev/test标签补训练或重新编码全湖来补缺口。

### 2.2 只读补评输入

补评R18同终点：A0 epoch2 SHA `a6658811a883973424a672cd0414685281caa2cb95e5fe163c884edc4e3650d0`；A1 epoch1 SHA `751fc997b2df3645326620d2c4e0518e570d46a8a063d3de8abe697892d155b1`；A2 epoch1 SHA `f403cf3f5cb1b40f6416331e262c26807587fb47e7480405542bcd3ecdbef87f`。已有selected排名无需重跑，但需核对hash和相同候选身份。

原B13、旧Teacher的逐query参考从R16 `candidate_pools.jsonl.gz`、`teacher_rerank_per_query.jsonl.gz`、`QT_RESULTS.json`读取并打包。B13只需同池Top50即可复算K≤50，不得用另一candidate set的29%代替。

## 3. P0：修复可复现性、补评和只读机制诊断

### P0.1 先设种子，再创建任何新参数

明确区分 `init_seed`、`train_seed`、`mining_seed`、`bootstrap_seed`。任何模型构造、权重扩展、DataLoader或随机扰动之前设置相应seed。保存父权重SHA、初始化后state_dict内容hash、批次ID序列hash及新参数hash。两个独立进程以同配置构造初始模型，应得到相同内容hash。

本轮C0/C1/C2/C3均以**同一个已保存A1父checkpoint**起步；所有臂AdamW重新初始化，不能称为继承旧optimizer的精确resume，因为R18checkpoint没有保存optimizer/RNG。主运行保存optimizer、RNG、sampler state以支持后续恢复。

### P0.2 全局分支调用必须一致，不能静默走旧接口

R18 A1只覆盖 `score_pairs`，继承的 `score_compressed_pairs` 不包含global residual。R18主evaluate实际调用 `score_pairs`，所以既有主排名不受这个风险影响；但旧R16 scorer或新cached reranker可能误走compressed-only接口。

实施其一：统一所有A1/C1调用到可取得z的pair scorer；或实现显式接受全局向量的compressed+global接口。对于A1/C1，在缺global输入时调用compressed-only接口必须清楚报错，不允许悄悄退回旧Teacher。

五关系均测试：batch/single、首次cache miss、真实第二次cache hit、checkpoint切换cache失效、save/load后输出一致。cache归属至少绑定checkpoint SHA、feature版本、object ID、表示类型、dtype、global_dim；不要跨臂共享参数依赖缓存。

### P0.3 补齐固定update的旧结构比较

只读评估上述缺失R18 checkpoint在U、D100、M上的完整排序，保存epoch1/2的自然曲线。报告：

- A1@5268−A0@5268；A1@10536−A0@10536；
- A2@5268−A0@5268；A2@10536−A0@10536；
- 原先的dev-selected比较继续保留，不能覆写旧结果。

这不是在旧测试上重新挑一个最好checkpoint，只是补固定终点解释。若A0终点意外很强，明确修订“续训无效”的边界，不静默改R19父权重。

### P0.4 冻结A1，无训练的分支诊断

在同一U上输出四种诊断score：

1. Full：`h(v_local + delta(z_a,z_b))`；
2. Global-off：`h(v_local)`；
3. Local-off：`h(delta(z_a,z_b))`；
4. Global-permuted：保持H、candidate ID、relation完全不变，只对z做同模态、全局固定、无自映射的对象置换。

置换用固定mining_seed=190911，映射独立于标签/排名且跨query一致，不能每个batch重抽；全局global cache需单独命名避免拿到未置换z的缓存。报告全query指标、positive ranks、分关系的RMS(local)、RMS(delta)、两者比例及score差异。置换只做一次，不扫到最差结果。

这些是**依赖性诊断，不是重新训练的架构消融**。Global-off可能因共同适应下降；Local-off好不能证明重训global-only一定不需要local；permuted下降支持语义对齐信息被使用，但不能独立排除额外容量效应。任何诊断不满意，都不能自动取消主训练臂。

## 4. 四个主训练臂与统一设置

| Arm | 父checkpoint | local_dim | global_dim | TT负例 / 列表 | 其余四关系 | 主比较与唯一干预 |
|---|---|---:|---:|---|---|---|
| **C0 A1 continuation** | 同一R18 A1@epoch2 | 512 | 512 | 原局部列表，原长度 | 原列表 | 共同续训基线 |
| **C1 Global-1024** | 同一父权重，经函数保持扩展 | 512 | 1024 | 与C0逐列表完全相同 | 与C0完全相同 | C1−C0：全局通路维度 |
| **C2 Natural-Hard-TT-Short** | 同一父权重 | 512 | 512 | 静态自然 hard reservoir，原长度 | 与C0完全相同 | C2−C0：TT competitor来源 |
| **C3 Natural-Hard-TT-Long32** | 同一父权重 | 512 | 512 | 同一reservoir；保留C2列表并追加至总长32 | 与C0完全相同 | C3−C2：同来源下TT列表长度 |

预注册主对照为 **C1−C0、C2−C0、C3−C2**。C3−C0用于总效果呈现，不能把来源与数量的联合变化归给单一因素；C1−C2/C3也不是单因素对照。R18父模型作为“训练0步”固定参考。

共同训练：2 epochs；训练列表仍42,143（C3只增长TT列表内部，不增加anchor/list数）；逻辑batch=8个lists；不跨逻辑batch再累积（原gradient_accumulation=1）；物理microbatch显存策略见第6A.3节；每epoch5,268 updates，总10,536；AdamW，LR=5e−5，weight_decay=0.01；dropout沿父配置0.1；无额外scheduler、gradient clipping、temperature、BCE、KD、confidence transform、loss weighting。保留五关系原频率，不新增relation balancing。

**固定 continuation train seeds = [13, 29]。** 两个seed均完成同样四个臂，不能看到哪个seed好才保留。若某臂因实际资源/数据缺口不能运行，保留其他结果并明确partial；不得以增加第三、第四个seed寻求显著性。

两个seed共享冻结的 hard reservoir、C2短列表及C3长列表manifest；每seed的C0/C1/C2/C3使用相同批次列表顺序。C2与C3的静态采样嵌套关系由第6.2/6A节规定。global widening扰动使用固定init_seed=190911，跨两个continuation seeds相同。该设计测试“给定同一A1父模型后”的增量鲁棒性，**不等于从R11重训A1的完整初始化种子复现实验**。

checkpoint保存：step0、5268、10536；主结果固定10536，不用edge局部Hit@1挑主模型。edge dev和natural dev在各点均报告，但不改变预算或主选点。新增中间checkpoint不允许事后最佳点替代主终点。

## 5. C1：只扩global到1024，局部Transformer保持512

### 5.1 具体维度

父模型：D=4096；local_dim=d=512；global_dim=g=512；global hidden=512；relation embedding=512。C1仅改g=1024：

\[
g_x=LN_g(G_{\tau(x)}z_x),\quad G_\tau\in\mathbb R^{g\times4096}.
\]

\[
r_g=[g_a,g_b,g_a\odot g_b,|g_a-g_b|,e_{ab}]
\in\mathbb R^{4g+512}.
\]

\[
\Delta v=W_2\operatorname{GELU}(W_1r_g+b_1)+b_2\in\mathbb R^{512},
\quad s=h(v_{local}+\Delta v).
\]

C1的W1为512×4608；W2仍512×512。保留num_heads=8、num_layers=3、FFN/local widths、16/24 learned latents、table group pooling和最终head不变。**不把type-pair embedding顺带扩到1024，不扩大W1的输出hidden维度。**

参数核算：C0=26,017,281；C1=33,361,921；新增7,344,640（约+28.23%）。整网model_dim=1024的原A1为78,773,249，约是现有A1的3.03倍，本轮不运行。

### 5.2 从已训练A1做函数保持扩展，不能直接随机重启

不能把旧512权重以`strict=False`吞掉shape不匹配；也不能拼零维度后声称LayerNorm不变。采用成对复制：

\[
G'_\tau=[G_\tau;G_\tau],\quad b'_\tau=[b_\tau;b_\tau],
\quad \gamma'=[\gamma;\gamma],\quad\beta'=[\beta;\beta].
\]

LN epsilon保持原值，因此数学上 \(g'_x=[g_x;g_x]\)。把旧W1分成5个512列块：前四块分别对应ga、gb、product、abs，第五块为type pair。前四块各扩成：

\[
[\tfrac12 W_i+E_i,\ \tfrac12 W_i-E_i],
\quad E_i\sim\mathcal N(0,(10^{-3}RMS(W_i))^2).
\]

第五块不变，bias1、W2、bias2和local/head全部复制。

成对扰动在step0乘以重复输入时严格抵消，并打破复制通道的训练对称性；不能仅复制且长期让两半完全相同，否则新增维度可能无法发挥自由度。固定扰动比例1e−3，不扫幅度。

数学上此迁移保持整个A1函数，而不只是恢复旧A0。本次审查用实际A1权重和合成五关系特征做了CPU验证，最大分数差约1.91e−6，复制投影两半获得不同梯度；**这不是实际数据精度实验，也不代替服务端真实feature验收**。参考脚本 `scripts/check_width_transfer.py` 是审查原型，不是已集成的训练入口。

验收：五关系真实输入每关系至少100 pair，FP32/eval下目标max_abs≤1e−5；1e−5至1e−4记录数值原因并检查cutoff处排名，>1e−4或系统性非平局换序停止C1。对C0/C1相同batch做少量update，确认两半G开始分化、梯度覆盖且无NaN。smoke实例丢弃，主训练重新加载父权重并迁移。

### 5.3 checkpoint和检索身份

新增明确的 `global_dim` 与 `global_hidden_dim` config；loader应支持旧R18的缺省global_dim=model_dim，也能strict-load C1。单测覆盖save→load→score，不能加载C1后悄悄按512重建。

Teacher仍是pairwise scorer，不获得ANN可分解性；Student 1024维及ANN不变。目标object的g可离线独立缓存，但pair交互仍在线。local token cache尺寸不变，global cache向量尺寸翻倍。新参数训练后需要重建对应learned caches。

### 5.4 结果解释

C1显著优于同预算C0：支持“在当前local512架构和训练设置下，扩大global通路有价值”，不能写成“所有Teacher必须比Student维度更大”。C1≈C0：削弱这个特定global扩容方向，不证明full Transformer1024必然无效。C1下降：检查函数迁移/复制对称性/漂移；correctness正常时保留负结果，不追加width/depth/LR扫参。

## 6. C2：显式的静态困难负例挖掘，只替换TT负例

### 6.0 当前到底跑过哪一种困难负例阶段

核查依据是本次上传 `src/run_stage1_r18.py`，而不是泛指仓库存在什么功能。R18 `train()` 加载固定 R12 `edge_lists.train_fit.jsonl`，每epoch仅shuffle同一批lists，再调用 `score_edge_batch`；没有在这个训练循环内调用ANN挖掘、重选candidate IDs或按epoch刷新hard negatives。**因此可以确定：R18 A0/A1/A2没有对新Global Teacher进行在线/周期性困难负例重挖掘。**

但不能写成“项目从来没有hard negatives”。完整仓库存在 `refresh_stage1_hard_negatives.py`、`run_stage1_rounds.py` 以及 `mmdd_stage1/mining.py`；旧通用流程还包含Student训练、重建Student索引、Teacher软标签和KD，不能直接整套启动来实现本轮Teacher-only实验。R11 supervision builder源码还包含语义相似、结构匹配、corrupted-path和random候选；这些可构成构造式难例，但不等同于“当前B13召回 + 当前A1最容易误判”的负例。R12审计源码检查R11→R12的candidate IDs不变，仅修改部分label。仅凭现有源码不能证明所有历史任务均实际执行过；执行日志/manifest缺失处保持unverified。

原v1第6.2节已经是静态hard-negative mining，本次不是另加一个同义训练阶段，而是澄清并验收它确实进入C2/C3训练。保存原列表negative provenance（可解析则分semantic/structure/corrupted/random/ANN/unknown），并把“仓库支持”“本轮调用”“真实候选变更”分开记。

### 6.1 本实验只改变负例身份，不改变列表长度

R18 TT train共12,041 lists、56,020 candidates，平均4.65 candidates/list；natural U平均289.37。局部dev指标实际是五关系macro **list Hit@1**，并非主query-macro Recall@10。高局部指标不能代替自然排序验证。

C2不改变list长度；新增C3独立测试32，不直接从约5扩到289，也不扫描多个长度。C2与C0逐列表保持：source、relation、positives集合与位置、negative slot数、candidate总数、list顺序、更新次数完全一致；只替换TT的negative slot ID。其余四关系逐字节保持原列表。

### 6.2 train-only自然池与静态挖掘

1. 从训练列表读取TT anchors，按**训练侧** `(source_id, directed relation)` 建known-positive registry。先核对训练source groups与主评测groups的划分；不得将固定1,198 query及其qrels拿来训练。数据湖目标对象允许按既有transductive设定重复，但source-group标注隔离必须按已有协议检查并披露。
2. 对每个训练anchor，使用冻结B13和既有检索入口生成train自然U：direct100加一跳Q→E→T；全部预算、QE/ET destination types、path cutoff和聚合参数从B13当时manifest提取、原样冻结。不得复制1,198个dev候选池冒充训练候选。
3. 仅使用已有冻结z/H覆盖的候选供Teacher静态打分。先报告过滤前后候选数和覆盖率，不自动下载/重编码缺失feature。若覆盖限制严重，标记这是feature-covered自然池实验，不伪称完整自然池。**必须核对实际candidate set是未按旧fusion预先压到Top50的D100 ∪ evidence-target union。** 不允许沿用极低evidence权重的top-K截断结果再称“自然union”；保留各target direct/evidence/both来源与QE/ET path。不得改变主评测的既有U。
4. 用**冻结R18 A1父模型**对train U一次性打分。剔除训练known positives及无效ID，然后按`(-score, object_id)`取最多50个候选，形成静态hard-competitor reservoir。不用当前训练中的C0/C1/C2/C3动态更新挖掘。父Teacher的高分只决定competitor选择，**不构成真实负标签，也不是Teacher-to-Student蒸馏软标签**。
5. 对每条原TT列表，保留所有原positive slots；对它的hard reservoir生成一次固定随机排列，无放回取前n个（n=原negative slot数）填回原negative positions，形成C2。抽样键由mining_seed=190911、source_id、relation、原列表稳定ID组成，使用稳定hash派生seed，不用受进程影响的Python内置hash。C3保留C2全部候选，继续从同一排列尚未使用的候选追加；不让C2只取highest-ranked而C3取更低rank来混淆来源/难度。不同continuation seeds共享最终训练manifest，不按结果换样本。
6. reservoir不足原negative slot数时，先扩到该train U全部有效非正候选；仍不足时用原列表剩余有效负例补齐并记录fallback数量。固定候选流分层记录为：top50 hard随机排列 → 剩余自然池随机排列 → 剩余原列表负例；C2和C3使用同一候选流，去重、过滤train-known positives。C3只在前一层不够时使用后一层，必须报告新增负例的层级与父模型score分布，防止把难度变化误当纯数量效果。缺失整个自然池的anchor保留原负例并标记未干预，不删除query改变分母。报告有至少一个negative被替换的TT list占比、各list实际替换数与采样候选重叠。

如训练anchor无法调用既有query入口，或必要features根本不存在，明确阻塞C2相应部分，不静默换到GT-based过滤/全湖随机负例。C0/C1继续执行；C2若仍能运行，则结论必须标明实际可干预覆盖。绝不因verified negative为0阻塞。

### 6.3 标签与loss

known positive→positive；其他有效候选→assumed negative；padding/invalid→mask。保留原raw label来源，禁止伪装verified。若挖掘候选碰到同一训练关系known positive，不能当负例。不得用dev/test qrels修正训练标签。

所有臂使用同一 `positive_loss_mode=sum_probability`：

\[
L=\log\sum_{c\in P\cup N}\exp s_c-\log\sum_{p\in P}\exp s_p.
\]

更hard的unknown中假负例比例可能上升；这是该协议的风险，不是预先否定训练的理由。本轮不加PU、confidence weight或人工负标签门槛。正结果不能证明挖掘的unknown都是真负例；负结果也不能唯一归因到architecture capacity。

### 6.4 困难度、来源与实际接入验收

冻结父A1对local/natural competitor的score、positive margin、top-negative score与rank、候选重复率，验证负例来源实际改变，并检验是否更接近当前错误竞争者，不能因文件名含hard就默认真的更hard。对每个anchor定义margin为 `max_positive_score - max_assumed_negative_score`；多正例另报告min-positive margin及positive softmax mass分配，不能只靠最好一个positive掩盖其余正例。

最少验收：源/目标ID、训练split/source-group、Student checkpoint/索引hash、mining Teacher hash、retrieval budgets、path来源、feature覆盖、原始score、排序与采样seed、去重/known-positive过滤数量、list hash、训练实际读入hash。C2正例/位置/列表长度与C0一致，C3含C2全部候选；不能只产出hard文件但训练仍指向原列表。

训练后按同样对象复测，报告原U中正目标从11–50晋升Top10以及原Top10跌落的query数。只在训练侧按稳定ID预先选最多256个anchors，可于step5268只读重打分原train-U，统计相对父A1 top10/top50 presumed-hard集合的overlap，以及原reservoir对当前高分competitors的覆盖。该诊断只用于判断未来是否值得动态refresh；**本轮不把新分数/新候选回写训练manifest，不因此改训练预算。** 负例过时也不是必须加动态挖掘的结论。

C2>C0：支持“TT自然候选竞争者对当前排序有帮助”，不是QE/ET都修好了，也不是listwise损失本身被证明优越。C2≈C0且覆盖充分：该静态recipe未显示价值；不立即扫top20/100、混合比例或新loss。C2下降且correctness正常：保留结果，结合假负例与梯度/score漂移诊断，不宣布自然分布训练普遍无效。

## 6A. C3：同一困难负例来源，仅增加TT列表长度到32

### 6A.1 要区分的假说

C2检验“负例选谁”，C3−C2检验“同一来源一次比较多少”。当前是pairwise Teacher加listwise loss，不是把多个target拼成跨候选Transformer；增长list不改变score(Q,T)的定义，也不直接扩大ANN召回。它可能改善有限候选池中的相对排序，**不能仅凭固定U提升就宣称跨2k/20k/200k湖规模泛化已被证明**。

C3在C2数据构造与feature正确性通过后即运行，不要求先看到C2显著改善。仅设一个新长度32；16/64/128/289都不跑。32是预先固定的中等计算规模探针，不是已证实最优值。

### 6A.2 嵌套构造与不变量

对每条TT list i，用经过已知正例保护的C2列表长度L_i、正例集合P_i与负例集合N_i：

\[
L_i^{long}=\max(L_i,32),\qquad
N_i^{long}=N_i\cup A_i,\quad |A_i|=L_i^{long}-L_i.
\]

A_i只能来自第6.2节同一候选流中尚未使用的有效assumed negatives。不删除、不重排C2原候选，不修改positive slots；原列表超过32则保留原长度。追加candidate IDs必须唯一、同relation类型、features可用且非train-known positive。训练侧保护排除已知正例，不用开发/测试qrels纠错。

如果候选流耗尽，使用能取得的真实长度，记录`length_partial`，不能重复同一negative凑32、不能把padding当负例、不能删除该list/query，也不自动换成全湖随机填充。全轮报告请求/实际长度的min、p50、p95、max及达到32的TT比例；很多lists未增长则只能得出partial exposure结论。缺少verified negative不是问题。

C3其余四关系的candidate IDs、labels、顺序逐字节保持C2/C0一致。C3不是“所有模态list一起扩长”，不引入in-batch negatives、loss temperature、relation weighting或BCE。

### 6A.3 不能只改一个长度配置

R18调用的`score_edge_batch`仅遍历各`example.candidate_ids`，再按原列表切分scores；普通batch增加不会把其他query的targets自动加入当前softmax。`list_truncation=null`也没有替你补足负例。因此必须生成并实际消费`train_negative_manifest.natural_tt_long32.jsonl.gz`，检查模型收到的有效candidate_mask数量确实增长。

保持现有完整list损失：

\[
L_i=\operatorname{LSE}_{c\in P_i\cup N_i}\,s(q_i,c)
-\operatorname{LSE}_{p\in P_i}\,s(q_i,p),\quad L_{batch}=\frac1B\sum_iL_i.
\]

不能把32候选拆成多个独立小list分别softmax后求平均；那不是同一个实验。不能因为long-list原始loss更高就判训练变差：分母项变多，本来就可能提高起始loss。用相同冻结诊断池上的margin、正例排名及自然R@10比较效果。

显存不足时可按完整list做microbatch，按实际逻辑batch的list数加权累积梯度，仍每8个lists执行一次optimizer.step（最后不足8按实际数）。五关系频率、总updates、LR/weight decay和完整list softmax保持不变。优先整list微批，不在pair分块时detach/no_grad丢梯度；若必须pair分块，须先证明保留完整分母与全路径梯度。microbatch规则在四臂一致可用，并验收FP32/dropout-off下损失及梯度等价；不能声称dropout下逐bit训练完全相同。

### 6A.4 更新预算相同，不是计算预算相同

C3仍2 epochs、10,536 optimizer updates、相同逻辑list顺序。长list导致更多pair评分/反传，是本次干预的一部分；不能称等算力因果优势。

按R18汇总：全关系每epoch164,449 pair slots，其中TT为56,020、12,041 lists；若每个TT原长不超过32且均成功扩到32，C3每epoch为：

\[
164449-56020+12041\times32=493741,
\]

约为原来的3.00倍pair slots（不是GPU时间或FLOPs的实测倍数）。原TT存在超过32、coverage不足或重复feature时，实际数依manifest重算。分别报告逻辑pair exposure、unique pairs、GPU-hours、峰值显存、缓存构建与挖掘成本；固定末步为主，不临时给C2多跑三倍steps“配平”，也不靠early-best替代末步。

C3>C2支持“同来源下更多负例竞争及相应额外训练暴露有用”，不能排除更多计算/更多distinct negatives本身的贡献。是否等算力仍然占优留待有正结果后单独匹配预算，不在本轮追加更多控制臂。

### 6A.5 结果解释

- C2>C0且C3≈C2：来源/困难度更重要，32未显示额外价值。
- C2≈C0但C3>C2：短hard列表可能不足；支持继续研究数量/竞争覆盖，不需要先让C2通过精度gate。
- C2>C0且C3>C2：两个阶段干预均有增量，不表明它们与1024扩容一定可叠加。
- C3下降：首先核查nesting、known-positive保护、padding/mask、完整softmax、训练预算与梯度；无bug则保留负结果，结合多正例覆盖、假负例风险及梯度集中诊断，不能继续扫长度或把unknown改回人工标注门槛。

negative diagnostic只使用已有标签来源/同train关系known-positive冲突，不新建PU、LLM负标签审核或downweight配方。

## 7. 统一评估与统计

### 7.1 固定三个池，不能用候选增加解释排序提升

每个checkpoint在同一U、原ANN D100、同数量M上打分；每个query四臂pool成员完全一致，U与M大小相等，重叠target分数完全一致。原qrels不改。U的RawRecall必须恒定约69.6717%；reranker不能提高固定U的RawRecall。

主要端点：全query-macro R@10，固定末步。次端点：R@20、CR@50；implicit/explicit；按source_group成对bootstrap10,000次；W/L/T。报告U−M，同一个scorer同一个checkpoint，不混不同模型。

### 7.2 两个seed如何汇总

每seed独立报告C1−C0、C2−C0、C3−C2及CI。主汇总先在同一query上对两个seed的Recall求均值，再按source groups bootstrap，不能把2×1198条当作独立query。另报告两个seed增量的min/mean/max；两个seed不足以精确估计训练种子总体方差。三个预注册主对照全部报告点估计、CI和W/L/T，不只挑最正者；这些是开发性比较，未经额外校正的区间不声称全族错误率受控。

作为本轮继续推进依据：主R@10均值正、配对CI下界>0，且两个seed的增量同号；同时披露CR@50、implicit/explicit的实际变化。CI跨0标记inconclusive，不靠加seed到显著。满足统计条件但幅度很小或成本显著增长，仍应按gain/cost判断，不机械宣布上线。

### 7.3 B13对照

补齐逐query B13在同U的Top50，独立计算C0/C1/C2/C3对B13的R@10/R@20/CR@50及CI。R18只提供A1对B13 R@10的bootstrap，不能据此把R@20的差距或CR@50领先也叫作显著。

主评测集合是开发证据；若已有独立test split，维持封存，不在本轮用它选C1/C2/C3。后续论文测试须在方法和预算冻结后执行。

### 7.4 来源与排名带

每个positive输出：D100、exact-D100、M、U、evidence-introduced、各模型rank、Top10/20/50 flags。membership从原候选ID复算；exact-D100原始列表不可用时标记unverified，不据其他字段猜。

来源分both/U-only/M-only/neither。必须同时报告：全query分母下的贡献分解（主解释）和positive pair数量（机制诊断）。不能用121个U-only子集结果代替全部query主指标。

对in-U positives统计1–10、11–20、21–50、>50、absent；特别报告原A1的365个rank11–50正目标如何变化。统计Top1/Top10 target集中度与候选可用频次，hub下降仅作诊断，不视为已证明根因。

### 7.5 QE/ET及图像边界

至少复算五关系edge-dev指标并使用正确名称relation-macro list Hit@1；image→table必须单独报告（R18 A1 88.6%，A0 90.8%）。保存逐list预测和标签，不只输出汇总。

若已有冻结Student QE/ET自然池及positive/witness字段，可做只读重打分，分别只替QE、只替ET，避免同时替两个edge却指认某一个根因。缺少原始pool/labels时记not_run，不额外构造新benchmark，也不阻塞主QT实验。不得把局部2候选QE准确率写成Teacher全湖evidence Recall。

### 7.6 Fusion：增加低成本诊断，不训练自适应RRF

**当前事实：** R18 natural-union主排序是对`combined_score_candidate_ids`逐pair计算Teacher(Q,T)，然后分别在U/D100/M中降序排列。该主结果既没有equal-RRF也没有weighted-RRF。R19四个主臂继续相同规则。因此，不能把A1目前与Student的差距直接归因于RRF权重，也不能悄悄把fusion改了后再给Teacher结构记功。

用户拒绝的是“为了overall精度把evidence压到近零、使系统退化direct-only”的捷径，而不是原则上拒绝一切可学习的证据可靠性判断。但让query-level gate自适应地输出接近零，同样可能退化；仅把fixed改成learned并不保证保住evidence贡献。禁止fixed weight scan、weighted-RRF、GT implicit/explicit在线路由；本轮也不训练adaptive-RRF/gating网络。

**F0：复用固定池与已有edge/path产物的只读诊断。**

1. 核查主teacher scorer在第一次target top50/top10截断**之前**看到完整既有U，而不是低evidence权重融合后剩下的direct-like候选。单独记录历史path预算造成的生成限制，不改当前冻结评测池。
2. 对有冻结natural QE/ET路径及raw scores的query，从原manifest重建Student evidence-target rank；QE、ET、path聚合、重复target处理、path上限全部保持原规则。保存target来源及其实际paths，缺失score/manifest时`missing_inputs`，不重新挖评测池、不借GT路径补齐。
3. 可输出四份**诊断**：同U的B13-D score排序、Student-E排序、equal-RRF(k=60)、固定A1或各固定末步Teacher-QT排序。复用已有raw scores及已获授权的固定模型评估；不能只拿B13 Top50名次伪造全U raw scores，缺失则相应项记not_run。Student-E对无path的U内target使用缺失标记并排在有有效score的target之后，以固定ID打破缺失并列，不凭标签填充排名。主结果只认Teacher-QT。equal-RRF严格为`I_D/(60+r_D)+I_E/(60+r_E)`：r_D按原D100排名，r_E按原冻结evidence目标排名；不存在于某一路则该项为0，不把缺失伪填第101名。它测试原两路rank信息，不声称融合了Teacher内容验证；B13-D对全U的score排序是另一诊断，不混用它与r_D的定义。无edge原始分数则跳过该项，不重新扫fusion方案。
4. 按全query宏平均与U-only/M-only/both来源解释：evidence-only正目标是否在池内但被QT压低；equal-RRF是否仍“implicit上升/explicit下降”；QE-only和ET-only替换结果是否改变这个现象。只使用已有标注，保留unknown；known-witness命中不等于证据内容正确，更不是Stage2值恢复。
5. 任一诊断即使overall提升，若evidence-only/matched-M-outside正目标的保留大量消失，须明确披露其direct退化性质，不能直接作为本轮主方法或论文第一贡献证据。不能根据诊断事后选fusion，再重写主端点。

**未来路线（本轮`deferred_not_authorized`，不是必须执行的第五训练臂）：** 若Teacher QT继续压低一些有可信path支持的U-only目标，同时QE/ET固定池诊断显示可用区分信息，优先考虑target-conditioned证据验证`H(Q,T,E_{q,t})`，而不是只按query决定整路权重。未来需要同池QT控制、内容/路径打乱对照、step0等价的零初始化残差、来源分解和新增pair成本；无需也不能以Stage2通过作为触发门槛。本轮的F0只提供设计依据，不因“表现不好”自动启动新模型。

## 8. Stage1成本，不混入完整在线系统时间

本轮只报告Stage1候选重打分成本和训练/离线成本。Student ANN身份不变，不运行Stage2。

统一GPU型号（记录实际型号）、device/precision、pair批量、池及并发状态。R18 A1在cuda:1、A0/A2在cuda:0且存在并行运行；旧latency不宜作严格结构开销结论。

分开测：无缓存首次执行；destination缓存已建但新query；重复query全warm。记录candidate/pair数、cache命中与未命中、query feature读取/压缩/global投影/pair scorer/排序耗时，p50/p95、peak GPU、cache字节和build时间。

R18 p50计时从query feature读取之后开始，覆盖U与M的合并候选，不包括原始Qwen编码与Student ANN；global投影还对重复对象重复计算。不可把其66.6ms叫作端到端查询延迟，也不可把cache优化只施加C1再宣称扩宽更快。

有函数等价的cache加速时，统一应用四个臂并单独保存等价验收；不要把工程加速当方法精度贡献。没有Qwen新query encoding计时就明确missing，不补估计值充当实测。

## 9. 失败传播与停止规则

- mask、checkpoint、feature配对、seed生成、global绕过、stale cache等具体错误：先修相应路径并重跑受影响分数。错误只影响一个臂时不作废所有臂。
- tiny弱标签拟合不好：optimization_warning，不是自动停训门槛；只有非有限loss、梯度断路、参数不更新或标签mask错误才阻塞对应路径。
- 共用自然池/feature覆盖不足：C2/C3分别标blocked/partial，C0/C1照常；C2能跑短list不保证C3能达到32，实际长度不足单独报告。C2精度不显著不阻塞C3。
- C1函数迁移失败：C1 blocked_correctness，C0/C2/C3照常；不改为随机1024重训冒充同父函数对照。
- 全部实验完成仍没有增量：本轮停止，不加hidden/depth/LR/weight/RRF扫参。
- C1与C2都有效：下一轮才预注册二者组合以检验互补；本轮不得追加组合臂。
- C1有效且C2/C3无增量：优先global表示/交互路线；C2−C0有效而C3−C2无效：优先训练竞争者来源，不盲目增大list；C3−C2有效：优先竞争覆盖/数量与成本控制。C2短列表无效但C3有效不能被忽略。
- C3 OOM：先使用第6A.3的完整list微批规则；仍不能执行时`blocked_resource`，不能临时把32改成16后称同一实验。F0缺path产物只阻塞对应诊断，绝不阻塞主训练。
- 都无效但frozen local-off明显更好时，下一轮考虑global-only或global-local交互身份，先做匹配训练对照。任何fusion可学习扩展或动态hard refresh均需下一轮单独方案，本轮不追加。

任何结果均不触发Stage2或KD；Teacher达到可用水平后再另行确定KD问题，不能用Teacher改善自动宣称Student蒸馏成功。

## 10. 必须交付的产物和打包要求

输出建议目录 `work/stage1_optimization_r19_20260911/`，可改日期但不可覆盖R18。

```
PLAN_FROZEN.md / PLAN_FROZEN.json
INPUT_MANIFEST.json / CODE_HASH_MANIFEST.json
EXECUTION_MATRIX.json / COMPLETION_AUDIT.md / FAILURE_NOTES.md
P0/seed_reproducibility.json
P0/R18_epoch1_epoch2_rankings/*.jsonl.gz
P0/frozen_branch_ablation/*.jsonl.gz
P0/branch_statistics.json
training_protocol.json
train_negative_manifest.local.jsonl.gz
train_negative_manifest.natural_tt.jsonl.gz
train_negative_manifest.natural_tt_long32.jsonl.gz
train_hard_reservoir.jsonl.gz / train_candidate_source_manifest.json
natural_mining_audit.json / known_positive_protection_audit.json
negative_provenance_audit.json / nested_list_and_length_audit.json
hardness_and_staleness_diagnostics.json / train_pair_exposure.json
baseline_reference/{B13,T_old}_per_query.jsonl.gz
candidate_pools.jsonl.gz / qrels_and_source_groups.jsonl.gz
C0|C1|C2|C3/seed13|seed29/{config,initialization_hash,parameter_count}.json
C0|C1|C2|C3/seed13|seed29/{step0_equivalence,smoke_test}.json
C0|C1|C2|C3/seed13|seed29/train_history.jsonl
C0|C1|C2|C3/seed13|seed29/checkpoints/...
C0|C1|C2|C3/seed13|seed29/eval_step000000|005268|010536/*.jsonl.gz
PAIRED_COMPARISONS.json / PER_QUERY_WLT.csv
CANDIDATE_SOURCE_ANALYSIS.jsonl.gz / HUB_DIAGNOSTICS.json
EDGE_DEV_PER_LIST.jsonl.gz / STAGE1_COST.json
F0_FUSION_DIAGNOSTIC/{status,inputs,source_retention,metrics}.json
F0_FUSION_DIAGNOSTIC/rankings/*.jsonl.gz
RESULTS.md / CLAIM_BOUNDARIES.md
```

排名必须含query、source group、query_kind、positive IDs、候选IDs、raw scores、完整rank和指标；附B13与旧Teacher的原始逐query基线，避免下一轮只能相信摘要。打包实际运行的完整相关源码与tests（包括model/feature/scoring/loader/runner），以及所有阶段stdout/stderr。模型权重可以另包，但至少包含完整config、state-key/shape清单、参数量与checkpoint SHA。

**不要把“文件存在”当成完成。** Completion表要列planned / actually-run / status / gate / reason / usable-results；未执行、失败、blocked、partial、not_triggered分开。Stage2写`out_of_scope`。

## 11. 最终要回答的问题

1. R18 Global改进在相同终点上是否仍然成立？此前dev选择造成了多大差异？
2. 正确global信息是否被实际使用？local分支是否表现为有用、冗余或干扰（只读诊断边界内）？
3. C1的1024维global在同父函数、同训练数据和同更新预算下是否优于C0？
4. C2静态hard mining只改TT competitor来源能否改善Top10，而不是只继续抬高小列表Hit@1？实际训练是否消费了挖掘产物？
5. 哪个改动真正缩小与B13的Top10差距，同时保留U-only目标？
6. 准确率收益是否跨两个continuation seeds一致？成本是否合理？
7. C3相对C2只增长TT list是否有额外收益？新增pair exposure和GPU成本是多少？是否保护U-only/matched-budget-outside正目标？
8. F0是否显示候选被提前fusion截断、证据信息可用但QT未利用，或edge本身仍不可靠？当前是否有证据支持未来的target-conditioned verifier，而不是退化的query-level加权？
9. 本轮主要支持容量、竞争者来源、竞争者数量，还是三者均未得到支持？下一步只选择少量经证据支持的路线，不把未执行的fusion/动态mining写成负结果。

### 方法依据（非运行依赖）

- PyTorch 2.9 Reproducibility：在随机操作之前控制RNG；LayerNorm文档：归一化维度决定均值和方差。
- Chen, Goodfellow, Shlens, *Net2Net: Accelerating Learning via Knowledge Transfer*, arXiv:1511.05641：函数保持扩宽的一般思想。本文的global/LN映射是针对当前实现的具体推导，不能不经测试就称为任意Transformer通用扩宽。
- Xiong et al., *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval*, arXiv:2007.00808：自然检索竞争者与训练负例失配的背景依据，不构成MMDD中该假说已成立的证据。

- Ren et al., *RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking*, arXiv:2110.07367：listwise训练及候选多样性的相关背景，不意味着当前MMDD也必须做动态双向蒸馏。
- Qu et al., *RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering*, arXiv:2010.08191：更多竞争者、hard negatives与unlabeled positives风险的背景；本轮不照搬其去噪模型或把verified negatives设成gate。
