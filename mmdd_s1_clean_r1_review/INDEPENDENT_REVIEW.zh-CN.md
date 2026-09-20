# MMDD Stage1 CLEAN-R1 独立审计：2026-09-18

## 结论

本轮不是一个已被严格实现并验证的 CLEAN-R1 负结果。交付源码存在会改变训练内容、标签和检索公式的错误。最重要的是：Teacher 根本没有接受自然多 evidence bundle 训练；预取转换丢掉空支持正集；难例刷新未执行预定的候选范围和 C 任务。Student 另有 E 任务路由错误和训练/索引公式不一致。建议暂停 Teacher 追加训练、S-SUP、S-KD，先修复与验收，不能只让 KD 暂停。

原报告的候选覆盖不足是实际现象，但“Teacher三类能力均已学好”“P/J目标错位已排除”“自然证据供给不是瓶颈”的强结论不成立。现在还不能把坏结果归结为三元交互结构本身失败。

## 1. 本次实际做了什么

- 解压用户提供的 mmdd_s1_audit_20260918.tar.gz；72个manifest条目全部校验成功，14个生产源码模块与审计清单hash一致。
- 使用 dev.population、六轮 Top50 排名、五种完整C100排序独立复算；1198个query，1000个source group，所有视图同池，最大数值差2.78e-17。
- 用packet_health中的真实Teacher logits复算128个D、60个E、60个C固定面板的成对准确率和Top-k召回。
- 用上传的生产类执行小型合成探针，复现B上下文缺失、空override丢失、刷新缺项、Student公式/梯度路由错误。
- 没有实际训练、没有修改用户生产代码、没有访问test、没有运行Qwen。
- 包内不含冻结特征、实际raw原文/图片、全部checkpoint和逐query训练packet。不能独立重跑真实Teacher前向，也不能从本包定位raw embedding低覆盖的全部原因。源代码缺陷是可复现事实，完整训练每一步的影响数量仍需真实packet日志确认。

## 2. 同一C100的结果

| 排序 | Overall R@10 | Implicit R@10 | Explicit R@10 | Overall R@50 |
|---|---:|---:|---:|---:|
| RAW-QT-on-C | 10.55% | 7.40% | 13.69% | 17.40% |
| admission | 9.67% | 7.15% | 12.19% | 16.41% |
| J-natural-on-C | 6.26% | 5.84% | 6.68% | 17.92% |
| J-empty-on-C | 2.75% | 2.00% | 3.51% | 19.13% |
| P-QT-on-C | 2.59% | 1.67% | 3.51% | 19.05% |

C100 candidate recall/oracle R10=20.34%，949/1198 query无正例。D100的对应上限21.62%，931/1198无正例。C100不是D100的超集；它丢掉39个D100正对，引入21个D100外正对。

J-natural对RAW-QT差-4.2849pp，按1000个source group有放回、保留重复权重、query macro的10000次paired bootstrap，95%区间[-6.09,-2.43]pp。这是单次训练的dev不确定性，不是多seed稳定性证据。

原先11.64%是epoch3在D100上的数字，6.26%是同一模型在修正C100上的数字，不能把两者称为同池前后模型退化。epoch3只是在错误评测池下选中，修正池上是否仍最优未测。

## 3. 确定的Teacher训练缺陷

### T1. B packet没有接到自然B

train.py:546和632均传q_rank=None；builder:273–275将其转换为空列表。自然view输入空B；增强view只从空列表加入一条GT witness。因此正常训练的J输入只出现0或1条evidence，线上却一次读最多20条。不能把这次训练称为成功执行了bundle-level training。

修复是接入本轮固定raw train query rankings，在启动训练前验证每query的自然bundle，include_bundle时缺q_rank必须硬错误，不能静默退成[]。固定raw B不得随Teacher难例刷新改变。

### T2. 空正例override被预取吞掉

builder正确产生`positive_ids=[]`；_compact_packets保存它；_expand_packets用truthiness判断，[]被当作“没有override”。teacher_rank_term随后回退到D列表全部正例。

合成复现：自然空B的implicit样本原应skip，转换后loss=3.4657359，正目标梯度=-0.96875。这个梯度是无效支持ranking监督；L_cal还对同一implicit空上下文设置支持=0。相对排序与绝对校准不一定数学上不可同时满足，但这种mask丢失明确改变了监督。

正式回执显示Teacher使用prefetch-workers=4；而workers=1分支也执行同一compact/expand，因此单进程对多进程相等不能证明与原始语义相等。日志epoch3–6没有B空正集skip与该错误吻合。

修复必须区分None和[]，同时保留excluded_ids等mask字段。差分验收比较context、label、mask、skip、loss、梯度，不只是候选ID。

### T3. 所谓难例刷新漏掉了关键分布

refresh_hard_ranks中anchor_hard_pool(q)只返回原D榜；D的并集退化为rawQT128和自身的并集，没有自然evidence引入的目标。refreshed['anchor']从未填充，没有J-single重评分，self.anchor_rank也不更新。

源码探针确认刷新只调用三个P任务，anchor数量0；不能说“训练完全按计划，只有dev池接错”。C的真正刷新必须以(q,e)为key，因为J(Q,T;{E})依赖Q。

### T4. 模态/来源交替排序被随后全局按分数重排

merge_modality_ranks先交替text/image（或rawET和QT），MakeList.hard_pool又按score重排。前期会比较不同来源原始分数；后期C还可能比较未刷新rawET内积和已刷新Teacher-P logits。

合成例子期望[text0,image0,text1,image1]，实际变成[image0,image1,text0,text1]。对已有名次流应保留顺序；经过同一scorer重打分的列表才按该scorer统一排序，不能把两类语义混用。

## 4. 候选竞争诊断比原报告更具体

定义E-only=C100\D100，仅用于诊断，不能线上读取GT路由。

| 排序 | Top10中E-only比例 |
|---|---:|
| P-QT（不读取E） | 88.57% |
| J-empty（不读取E） | 86.49% |
| J-natural | 67.61% |
| 原始admission | 30.51% |

E-only占C100候选39.31%，只贡献21个正对。P不读取E却偏向这些候选最严重，故不能单凭“J读取evidence所以偏好evidence相似目标”解释。

额外诊断：统一限制到每query相同的C100∩D100集合，不增加任何正例、不用GT过滤候选，再保留原始相对顺序：P-QT R10=14.64%，J-empty=14.71%，J-natural=12.34%，RAW-QT=10.55%。这不是可与完整C100主结果直接混比的新系统，而是说明Teacher在direct-style竞争中并非完全不会排序，突出问题是跨候选来源的高位排序失稳。T3漏挖两跳自然目标是与此吻合的具体机制，但修复收益仍要实测。

D面板按未作为模型输入的ID来源前缀分层：Teacher对dl_raw_*竞争项的pair accuracy=99.14%，对target_*竞争项=63.36%；C分别99.25%和82.38%。这提示大池成对准确率部分来自容易的来源类型区分，不能证明模型在读ID或已有泄漏，需要真实内容/特征才能进一步归因。

## 5. 原报告把“学到区分”放大成“能力已健康”

| 面板 | query数 | 平均候选 | Teacher pair accuracy | Teacher R@10 | 平均正例名次 |
|---|---:|---:|---:|---:|---:|
| D | 128 | 924.84 | 77.12% | 10.94% | 213.20 |
| E | 60 | 257.55 | 79.28% | 25.28% | 54.92 |
| C | 60 | 227.07（mask后） | 89.88% | 56.67% | 25.00 |

这些列表插入GT正例，不是自然召回，也不是训练的31负例列表。C确有可用的局部区分信号，但D的大池平均pair accuracy不能当作“蒸馏已合格”。单正例时pair accuracy约等于1-(rank-1)/n_negative，rank两百多仍可得到约0.77。

P/J错位并未被这种面板排除；同样目前也不能断言错位是主要根因。必须先修实现再做匹配输入的机制判断。

## 6. 证据供应：局部82.9%不能外推全局

已入C100的implicit正对111个，其中92个在自然B中有已标witness（82.9%），J-natural在这92个中Top10命中33个，RAW-QT命中46个。因此入池部分不是单纯“无证据可读”。

全dev的678个implicit正对中，340个有已标witness在B中，其中248个目标不在C100。可见“找到了证据，但对应目标没进入最终小池”是重要的另一段损失。包内未交付完整L_E和U，不能把这248个再精确拆成第二跳未召回与admission丢失。

这里的visible是ID交集，不是重新核验原文/image内容及缓存张量有效性；未知witness不等于已证明无证据。

## 7. Student也存在阻断错误，SUP不能直接续跑

train.py:score_packet用`packet['mode']=='P'`区分任务，但D和E都是P，因此E实际调用query_direct，query_evidence没有任何训练packet的loss梯度路径（不含weight decay）。合成探针E的evidence.weight.grad为None，direct.weight有梯度。

同一函数把所有candidate keys改成base_e(nu_candidate)，且没有归一化；公式本来要求原始nu_candidate。挖掘又用unit(A_Dnu_T)、unit(A_Enu_E)，线上目标索引用unit(Bnu_T)。不是一个统一评分式。合成探针在有意设置非单位矩阵后D/E/C全都偏离公式，E排序可反转；D logit可超过归一化dot/0.07应有范围。

Student.refresh_mining同样没有填充anchor（C）排名。每轮开始生成keys，训练后却把这份旧keys送到下一轮refresh，而查询由训练后的参数计算，形成不同快照配对。

另外，先前回执显示SUP/KD都启动并在epoch1评测因缺少pipeline报错；交付代码补了pipeline，但返回{'ann': block}，run_split再套{'ann':...}，又形成嵌套。必须跑真正的train-one-epoch→dev→refresh集成验收，不能只测单函数。

这些不是Teacher低分的原因（Student在Teacher之后），但会使后续训练继续给出无效结论。

## 8. Query行修订未完全落实，不能盲目重建全湖

当前cache仍共用table_max_rows=12和七组行池化；models仍固定SLOT_COUNT=9。报告A7证明的是旧版“<=7逐行、>7合并”的行为，不是后来约定的Query全行独立保留。

不过本包只有全部table合并的行数统计，不能确定有多少query实际超过7/12。若全部query都只有5行，本轮该规则未触发Query信息损失；不能据此先重跑整个数据湖Qwen。先导出query-only原行数/保留行数/槽位映射，再按不兼容对象最小化重编码。

## 9. 建议下一步

暂停两条Student和Teacher追加训练。保留现有目录为原始异常run，不覆盖成绩、不宣布新架构已失败。修复T1/T2/T3/T4、评测池和Student公式；补端到端小型验收。核心干预会改变训练语义，不能从旧epoch3或Student epoch1当作未改变协议继续。

修复通过后，在新的run目录从本轮预定随机初始化重跑同一Teacher协议；符合指纹的冻结Qwen缓存可以复用。不要同时改网络宽度、层数、loss权重、训练epoch，避免把代码修复和方法搜索混在一起。

Teacher每轮必须同时输出同池raw与P/J指标、真实bundle长度分布、各任务正/负/masked计数，以及hard候选来源；下一次Teacher健康检查过关后才启动完整蒸馏。

详细修复合同见REPAIR_AND_REPLAY.md，源码定位见SOURCE_LOCATIONS.md，所有计算与探针输出保留在JSON中。
