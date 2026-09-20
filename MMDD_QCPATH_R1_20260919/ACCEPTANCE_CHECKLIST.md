# QCPATH-R1 合规与验收清单

每项必须对应实际证据。此清单是主文档编号索引，不替代完整规范。

| ID | 必须满足 | 执行证据 / 状态 |
|---|---|---|
| H01 | 冻结 Qwen3-VL-Embedding-8B；本轮不新增 Qwen 前向，不更新 backbone。 | 待服务器验证 |
| H02 | Student 每个目标一个静态向量；只允许 Q→T 与 Q→E→T。 | 待服务器验证 |
| H03 | A 只训练新增 adapter；冻结全部原 P/R、Direct、Q→E 和目标索引。 | 待服务器验证 |
| H04 | A 全目标竞争，不得用 Top32、in-batch-only、随机子集或 sampled softmax 替代。 | 待服务器验证 |
| H05 | A 残差相对上限 ρ=0.5；不是最终查询向量单位化。 | 待服务器验证 |
| H06 | Teacher 正式路径分数必须来自真实 Q/E/T 输入；QT-only 仅对照。 | 待服务器验证 |
| H07 | 不输入 GT join column、GT 恢复值、对象 ID/source ID/检索排名作为模型特征。 | 待服务器验证 |
| H08 | 在线不读取 implicit/explicit、witness 标注或正目标集合路由。 | 待服务器验证 |
| H09 | Unknown 是 assumed competitor，不称 confirmed negative；保护所有按本文定义的 train-known positives。 | 待服务器验证 |
| H10 | 不加载历史训练列表、路径训练图、负例列表或 Teacher 分数作为新增训练数据。 | 待服务器验证 |
| H11 | 不改已有 production Evidence 排序与 Equal admission；它们只分配候选预算，不代替最终 Teacher。 | 待服务器验证 |
| H12 | 不用 weighted RRF、Q-additive、QT+Path 插值、固定低 evidence 权重或手工提权救结果。 | 待服务器验证 |
| H13 | 不因缺特征删候选、删 query、删模态、复制 BASE 成两个 seed 或用假分数补齐。 | 待服务器验证 |
| H14 | 不重做 CLEAN-R1 的 P/J 双任务或固定八槽缓存；不增加多个 Teacher 主干/输出头。 | 待服务器验证 |
| H15 | 准确区分 planned / implemented / executed / evaluated；不得把测试通过称为真实训练完成。 | 待服务器验证 |
| H16 | 失败就按状态机停止；不得以“还没解释完原因”为由新开训练分支。 | 待服务器验证 |
| A-T01 | W2/b2=0时BASE/E-only/QE查询与raw score相同；真实向量probe同样检查。 | 待服务器验证 |
| A-T02 | 非对称非单位R下，逐pair、矩阵、索引公式一致；transpose错误必能被检测。 | 待服务器验证 |
| A-T03 | Δ=0初步梯度到W2非零；W1首步可为0，不能误判整个adapter断梯度。 | 待服务器验证 |
| A-T04 | 任意输入输出残差比≤0.5+1e-6；极大Δ、极小Δ、Δ=0均有限。 | 待服务器验证 |
| A-T05 | E-only替换Q后输出严格不变；QE替换Q在非零测试权重下有可测变化。 | 待服务器验证 |
| A-T06 | 全矩阵与chunked全分母的loss和adapter梯度一致；有块无正例也可计算。 | 待服务器验证 |
| A-T07 | ignore目标分数任意变化不影响loss，梯度为0；P不被忽略。 | 待服务器验证 |
| A-T08 | 旧Top32之外的合成高分competitor在新loss中有正的降分梯度。 | 待服务器验证 |
| A-T09 | empty P、empty N、标签重叠、不合法ID都有明确处理，不产生NaN继续。 | 待服务器验证 |
| A-T10 | microbatch累计等于同logical batch一次计算；尾batch不丢。 | 待服务器验证 |
| A-T11 | 每epoch每item恰好一次、两arm item/标签/权重顺序完全一致。 | 待服务器验证 |
| A-T12 | BASE全部参数与目标vector/index训练前后hash不变；optimizer白名单准确。 | 待服务器验证 |
| A-T13 | 缓存key包含q,e,modality,BASE,adapter版本；不同Q不能命中相同QE查询缓存。 | 待服务器验证 |
| B-T01 | e=EMPTY的step0 f0与原T0 QT真实输出近似相等（FP32 atol1e-5 rtol1e-5）。 | 待服务器验证 |
| B-T02 | g_QT全局表征分支没有被删掉；只有一个shared Transformer与shared head。 | 待服务器验证 |
| B-T03 | 实际triple forward读取三个对象；改变E内容可改变输出，改变ID不改变输出。 | 待服务器验证 |
| B-T04 | variable长度/padding/role边界正确；没有固定九槽假设，Query row组不丢。 | 待服务器验证 |
| B-T05 | Natural的输入路径来自真实检索，不是None/空数组统一回退。 | 待服务器验证 |
| B-T06 | Augmented同一个e*给同query所有candidate；负candidate也能看到它。 | 待服务器验证 |
| B-T07 | empty positive `[]`经prefetch和序列化保持空，不能恢复为G。 | 待服务器验证 |
| B-T08 | target正例不等于每条E都正；未知替换E不把正确T翻成负例。 | 待服务器验证 |
| B-T09 | path分块/target分块只分计算，不分softmax；与不分块loss/gradient一致。 | 待服务器验证 |
| B-T10 | multiplicity优化前后S=logsumexp相同；空evidence时S=f0。 | 待服务器验证 |
| B-T11 | QT-cont最终score没有加log(path_count)；主QET没有外部T0标量fallback。 | 待服务器验证 |
| B-T12 | 真实一次“取query→Natural/Aug→prefetch→三对象forward→LSE→loss→backward→step→评测”集成通过。 | 待服务器验证 |
| B-T13 | 冻结压缩缓存不包含任何可训练role/Transformer/head输出；跨checkpoint不混score。 | 待服务器验证 |
| B-T14 | E-swap不改变candidate IDs/路径槽数/m、模态或GT分母；不重检索。 | 待服务器验证 |

## 阶段交付

| 阶段 | 允许继续的条件 | 必交产物 |
|---|---|---|
| S0 | 原始输入与生产函数锁定，数学/生产/真实小样本测试通过 | RESOLVED_INPUTS、SOURCE_LOCK、PRODUCTION_CONTRACT、测试报告 |
| A13 | 主文档第7节全部条件通过 | 两臂epoch3、raw排名、P/I/N、A13 decision |
| B13 | 主文档第13节全部条件通过 | 两臂epoch2、同池QT/Path/E-swap、B13 decision |
| A29/B29 | 各自再次通过同一门槛 | 独立seed结果、共同池和own池分开汇总 |
| 结束 | 不再启动新训练 | RESULTS、LIMITATIONS、NEXT_DECISION、可复算交付包 |

## 状态严格区分

`PASS`=实际满足且有证据；`FAIL`=实际不满足；`BLOCKED`=缺必要输入/资源/存在矛盾；`NOT_REACHED`=先前门槛未过，未执行；`NOT_APPLICABLE`只用于主规范明确允许的不适用项。不得用后两者绕过必选测试。

```json
{
  "requirement_id": "A-T06",
  "required": true,
  "status": "NOT_REACHED",
  "evidence_path": null,
  "actual_value": null,
  "reason": "示例；不是本轮执行成功回执"
}
```

**执行前、代码优化后、正式训练后、最终打包前都必须重新核对相关清单。没有证据不能勾选PASS。**
