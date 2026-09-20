# 执行端验收清单

状态只能填 PASS / FAIL / BLOCKED / NOT_REACHED，并给实际证据路径。
本包的参考测试通过**不能**自动将下面全部标PASS。最后一列由执行端填写，不能只写“已按要求”。

| ID | 验收项 | 最小证据 | 状态/路径 |
|---|---|---|---|
| C01 | 原split及全部原train | SCHEMA_MAP / DATA_SPLIT_REPORT；无cal桶 | 未执行 |
| C02 | G/D/W从原GT重建 | 原记录定位、字段映射、label_stats | 未执行 |
| C03 | 无train/dev/test标签混用 | 标签扰动测试与loader边界 | 未执行 |
| C04 | 不存在旧任务产物根 | ROOT_INPUTS/READ_AUDIT/no_history测试 | 未执行 |
| C05 | 纯缓存来源与缺失生成 | ENCODER_CONTRACT/FEATURE_COVERAGE | 未执行 |
| C06 | 全部Query行保留 | query-only逐对象行数/摘要数 | 未执行 |
| C07 | PCA本轮重算 | PCA来源集合、basis与均值指纹 | 未执行 |
| C08 | Teacher全部参数新初始化 | INIT_LINEAGE、初值、未加载旧权重 | 未执行 |
| C09 | Teacher学习pooler和global | 真实前反向梯度、all-param白名单 | 未执行 |
| C10 | 一种Teacher架构、每模型一个共享head | pair/QET实际前向hook与输入e变化测试 | 未执行 |
| C11 | T_EDGE完整2epoch | 消费顺序、每q/关系计数、优化步数 | 未执行 |
| C12 | 一次真实难例刷新 | raw两路pool、刷新ID差、正例保护 | 未执行 |
| C13 | T_PATH完整2epoch | 真实自然/共享增强路径、loss/梯度 | 未执行 |
| C14 | 目标正集与路径正集未混淆 | G target loss；P/I/N conditional loss | 未执行 |
| C15 | SUP/KD新初始化及同C1输入 | 两个初值hash、相同候选/顺序 | 未执行 |
| C16 | C1完整1epoch | 五关系梯度、更新和active query计数 | 未执行 |
| C17 | 三个条件化C2冻结P/R和静态index | C1/C2前后tensor/index hash | 未执行 |
| C18 | QE/EONLY仅输入条件区别 | 同prefix/同adapter init、Qmask测试 | 未执行 |
| C19 | 完整分母与分块梯度 | 全合法target数量、跨chunk正例对照 | 未执行 |
| C20 | C2 target-path SUP/KD实际执行 | 分项loss、adapter梯度和teacher detach | 未执行 |
| C21 | 空P/ignore经过prefetch不变 | 真实序列化roundtrip及inactive计数 | 未执行 |
| C22 | 尾batch/microbatch一致 | loss/梯度与完整logical batch差分 | 未执行 |
| C23 | own ANN非旧池重排 | 各模型D/E/U/C/paths与query模型hash | 未执行 |
| C24 | 最终真实Path非QT补分 | 调用trace、LSE重算、外部QT未接入 | 未执行 |
| C25 | Raw/六Student及独立QT同协议 | own主表、候选表、D/U/M对照 | 未执行 |
| C26 | Q作用与E内容作用分开 | 固定probe、E-swap同槽相对排名 | 未执行 |
| C27 | strict双排除及失败分母 | raw IDs/qrels整数复算 | 未执行 |
| C28 | 固定终点和明确seed预算 | 每stage last与repeat gate；不挑峰值 | 未执行 |
| C29 | 原test只在配方锁定后评 | test读取记录、模型/seed数冻结hash | 未执行 |
| C30 | 数据与代码回执诚实 | 实际executed source、资源/错误、planned区别 | 未执行 |
| C31 | T_QT独立训练而非f0重命名 | 共同T_EDGE初值hash、两分支optimizer/训练回执 | 未执行 |
| C32 | T_QT target正集为G且无QET调用 | 隐式target正例测试、forward trace、loss项 | 未执行 |
| C33 | QT-only Student从新table参数开始 | init hash、无多模态任务parent、允许模块清单 | 未执行 |
| C34 | QT-only train数据无E/W依赖 | loader读取日志、候选生成来源、G scope | 未执行 |
| C35 | QT-C2完整分母与双侧P梯度 | 全矩阵/分块loss与P/R梯度对照 | 未执行 |
| C36 | KD-NATIVE无adapter且ET不读Q | state_dict、改变q的ET不变、P/R更新与新index | 未执行 |
| C37 | 固定池Teacher三种差值分开 | same C100 IDs、T_QT/T_PATH-f0/Real/swap分数 | 未执行 |
| C38 | 严格Direct-only与输入消融分开 | QE/ET调用0、独立QT模型和多模态D100不同表位 | 未执行 |
| C39 | 两卡并行与同卡共驻按合同 | PROFILE/SCHEDULE事件、reservation/吞吐/差分 | 未执行 |
| C40 | 并行RNG/optimizer/cache隔离 | 进程ID、GPU UUID、branch cache keys、原子发布 | 未执行 |
| C41 | 13阶段DAG/26预算与回执一致 | DAG拓扑、seed选择、实际stage/epoch/重算数 | 未执行 |
| C42 | 历史比较明确差异且不反流 | LEGACY_COMPARABILITY与references-only读取 | 未执行 |
| C43 | 正式时延独占并发吞吐另报 | 测量GPU同驻进程记录、warmup/timing范围 | 未执行 |

F01–F23的所有强制约束同时适用。缺少某阶段只能标NOT_REACHED或BLOCKED，不得用局部成功替代。
