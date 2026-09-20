# 迁移到v2.1：新增对照与调度，不改变fresh主线数值配方

## 尚未训练

使用新完整文档/JSON/DAG。复用原dataset/backbone位置和已验证纯内容缓存；不读取历史B13/T0、旧train列表、旧PCA和旧校准桶。旧QCPATH局部实验不继续启动。不要混用v2.0与v2.1配置。

## 同一个v2.0 fresh运行已经有进度

不要因为新增对照机械地删除或重跑全部进度，但也不直接把别的run最强权重放进来。

允许在**同一个已登记的fresh运行谱系**内形成v2.1协议追加记录：核对原root、初值、特征、监督、候选、loss、dtype、batch、RNG、更新顺序和阶段终点未改变；保留原stage执行版本及source hash，写 `VERSION_ADOPTION.json` 列出逐阶段等价结果和新的控制分支。主线数据namespace继续用v2前缀，新增对照用独立CONTROL namespace，不消耗原RNG。schema/代码更新不得改变旧阶段的数值行为。

只有全部未变的已完成stage可以作为这一运行自己的前缀。不能复制其它历史run，不能只凭权重文件名或相近分数判等价；不能复用同一阶段中途有实现错误的结果。

本轮T_EDGE已完成时可以从该冻结快照追加T_QT，不能用最终T_PATH代替T_EDGE。KD-C1已完成可以增加KD-NATIVE，不需要重复KD-C1。QT-SUP/QT-KD必须新初始化，不能接多模态Student已学参数。

若无法证明某已执行阶段与本版主线数值合同一致，明确列出差异/依赖范围；不安静接续，不自动无限重训。保留所有原产物和错误报告。

## 永久废止的旧要求

历史train_fit/cal桶、旧parent局部先行gate、旧QCPATH的8作业限制均不适用。v2.1本轮上限为13阶段/seed、最多26。GPU统一2×RTX4090；安全独立双卡及同卡并行按新调度章实施。

任何历史checkpoint只能出现在隔离references评测，不作为本版控制组的初始化。新增QT对照的存在也不授权把正式Path方法改成QT-only。
