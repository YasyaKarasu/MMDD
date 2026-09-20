# 执行 FRESH-PATH v2.1：完整fresh与独立QT/旧式Student对照

你必须完整阅读主文档、protocol.json、CONTROL_MATRIX、EXECUTION_DAG、PARALLEL_EXECUTION、验收清单和迁移说明后执行。不得只依据这段提示词自行设计。

## 当前用户要求

从原带GT数据集及原train/dev/test开始完整fresh Stage1；所有任务模型新初始化，训练组合在本轮产生。历史B13/T0、旧负例、旧Teacher分数、旧PCA和旧learned压缩缓存均不能作为训练依赖。没有历史train_fit/calibration桶。

主方法仍是KD-QE + 本轮T_PATH真实路径重排；同时必须训练独立T_QT、QT-SUP/QT-KD和KD-NATIVE。不能拿T_PATH的f0评测冒充训练T_QT，不能拿多模态Student关闭E入口冒充训练QT-only，不能拿带MLP的KD-EONLY冒充原无adapter结构。

## 必须执行的13阶段/seed

公共fresh T_EDGE两epoch后刷新一次难例，分别复制为T_PATH两epoch和T_QT两epoch。
主线S_SUP_C1/S_KD_C1各一epoch；SUP-QE/KD-QE/KD-EONLY C2各三epoch；新增KD-NATIVE从本轮KD-C1派生，原P/R无adapter训练一epoch。
独立QT-SUP与QT-KD各从新table PCA/identity开始，C1一epoch、C2三epoch。QT-KD只蒸馏T_QT的QT分数。
各stage依赖、参数范围、loss、正集和候选按专章执行；不能把“C2冻结P/R”错误套给NATIVE/QT-only，也不能把它们的P/R训练规则套回条件化主线。

主线C2的全目标SUP、条件KD、target-path SUP和KD一项不能漏。T_QT对implicit正确target仍用G作正例，不得只用D削弱对照。QT-only Student不读W/Epos或两路候选。

## 硬件与并行

整个项目从始至终为2×RTX4090。按DAG先利用两卡独立作业（如T_PATH/T_QT、SUP/KD、QE/EONLY），再按资源/正确性/吞吐profile允许同卡多进程。不要因为一个进程占一张卡，就说两卡不能并行。
每作业独立进程/随机状态/输出，严格保持batch、精度、样本顺序、更新数和缓存身份。不用DDP改变逻辑batch，不为共驻改AMP/量化/截断。CPUworker和LRU按全机预算分配。原子发布共享缓存，拒绝未完成上游的半文件。正式效率时延使用独占GPU窗口。

## 执行与止损

先完成P0/P1及新增控制组的真实集成验收；参考测试不代替服务器验收。之后直接执行已授权seed13完整主线及对照，不反复问是否继续，不先跑局部历史parent实验。首seed13阶段和表全部完成后才按固定gate决定seed29；最多26有效stage，无第三seed/额外反馈/超参搜索。

独立T_QT、同权重T_PATH-f0、T_PATH-Real、E-swap在同一C100比较。另报各Student own检索、固定T_QT下的候选收益、独立QT-only、旧式NATIVE+QT工作点以及兼容条件下历史参考。不要混淆候选差异和排序差异。

如v2.0同一fresh运行已有合规阶段，按MIGRATION说明逐阶段检查是否数值合同完全未改；不要为新增对照无故删数据/重跑，也不能借迁移挑任意历史最强模型。

必须逐项给验收证据，区分planned/implemented/executed/evaluated。确有数据/实现/资源障碍时报告具体项；不能自行删控制组、换parent、放宽门槛或把QT改成主方法。不要建设通用实验平台；完成实际数据管线、训练和评测。

**每项规划与约束均须遵守。完整fresh、真实Path主线、独立QT对照以及安全并行缺一不可；执行代码完成不等于训练完成。**
