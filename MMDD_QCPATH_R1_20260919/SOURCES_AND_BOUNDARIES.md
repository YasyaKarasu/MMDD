# 来源、设计选择与验证边界

## 一、来源支持的事实

### S1：`INTEGRATED_REVIEW.md` — Query-conditioned E→T 与 R30 综合复核，2026-09-16

从用户Library/项目上下文读取的历史审查。重点使用其“已完成的核心设计”“主指标”“巨大residual/公共目标”“正例保护范围”章节：原小MLP+零初始化+冻结BASE与目标库；14592固定列表；QE残差约48倍；939个verified pair共同exact Top1且该目标未进入训练候选；P/I标签范围应澄清。

这是历史复核记录，不是本次重新训练、加载旧adapter或全湖推理。本文没有把“候选遗漏”写成唯一根因。

### S2：用户上传 `MMDD_current_chat_transcript_20260919(1).md`

关注原用户约束和EXP3/CLEAN-R1审查：Teacher–Student职责、真实路径最终评分、保留Query每行、不建设多模型拼装系统、train GT可用但旧训练列表不能作为fresh起点、空正集序列化错误、自然bundle漏接、分数/索引接线不一致、拒绝无限诊断与救火。

其中EXP3已存在path训练，不能继续以“没训过path”解释全部失败。该文本是历史对话，含过去被修订的建议；不把所有段落同时作为现行规范。

### S3：用户上传 `BRIDGE_INDEPENDENT_REVIEW.md` 与 bridge材料

健康短C1与长C1存在差异，不能将C1@659当默认健康起点。本轮没有重跑bridge，没有将其汇总当作新增own-pool结果，也没有试图把步数356当普适定理。

### S4：实际读取的源码快照

来自用户上传 `MMDD_src_stage1_bridge_20260915.tar.gz`，仅为2026-09-15快照，不冒称服务器当前源码：

| 文件 | 本次检查的内容 | SHA256 |
|---|---|---|
| `src/run_stage1_r19.py` | R19GlobalResidualTeacher的局部REL＋全局关系表征、共享head、缓存身份 | `a8439137b587f945ffe2ea9ddd24d9c75d4777eeeacdd96ddf6f4052f7635822` |
| `src/mmdd_stage1/models.py` | 原Teacher配置/压缩与Student relation_query/index_vector行向量语义 | `2f59151b3c7194b14e34a4db31a2d0cdef55ee77e62ce997d2e1983f8beffffd` |
| `src/mmdd_stage1/b13_recipe.py` | 旧B13 SUP用10*sigmoid、KD用raw；新A不能误继承该变换 | `00688771844152107aa3fe0468e9133c4bfee29271e1f2fe37d3f571d0b16454` |

没有提供QC-ET最新源包本体，因此本次没有直接核验其每个函数；其14,592/939等数字引用S1记录。服务器必须通过输入/源码锁确认当前实现，不能只根据这些历史文字宣称一致。

## 二、本轮新增且尚未经实验验证的决定

全目标分母＋受限残差同时作为有效配方修正；ρ=.5；query-macro曝光权重；A/B硬门槛；单seed先行；B使用最多2048+2048 train query；保留原强T0全局表征的三对象兼容扩展；Natural/Augmented view；支持margin1、权重.2；epoch3/epoch2终点；有条件重复seed29。

这些是预定工作点，不是历史最佳或理论最优。它们已在主规范显式固定；执行者不得根据自己的偏好改掉。

## 三、特别澄清：与旧建议的不同

1. 旧综合审查曾建议先单独改全目标分母。本次最近对话已决定优先拿到可用配方，允许分母与残差约束一同修正；不能再把新旧实验差值解释为单个因素效应。
2. 上条对话的Teacher `F(Q,E,T)`是抽象式。此文落地时发现强T0有既有global residual表示，故明确保留它与共享head，而不是悄悄把强Teacher退化为局部-only弱模型。最终没有外部QT标量融合，E作用需要通过固定内容对照。
3. 本轮是局部可行性验证，不是fresh重建。新增列表重新生成不等于权重也fresh。最终fresh与KD须下一轮独立规范，本包未授权执行。

## 四、本交付实际完成的验证

见 `AUTHOR_VALIDATION.md` 与 `reference/pytest_output.txt`。只包含CPU数学/规则参考测试和文件一致性检查；没有用户服务器上的Qwen、B13、T0、真实ANN或GPU训练。
