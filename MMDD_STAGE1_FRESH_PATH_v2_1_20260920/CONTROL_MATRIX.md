# 对照组定义与唯一解释

## Teacher

| 名称 | 从哪里来 | 训练期间有E信息吗 | target重排读E吗 | 正确用途 |
|---|---|---|---|---|
| T_EDGE-QT | 本轮fresh edge终点 | 有，五关系pair | 否 | 旧式pair-only中间参考，不冒充最终QT最优 |
| T_QT | 从同一T_EDGE分叉，独立QT target训练 | edge warm-up/replay有；没有QET path训练 | 否 | 独立训练的QT-only重排对照 |
| T_PATH-f0 | 本轮T_PATH权重不变 | 有 | 否 | 输入移除对照，不是独立QT训练 |
| T_PATH-Real | 本轮T_PATH | 有 | 是 | 主方法 |
| T_PATH-E-swap | 与上一行完全同权重/候选/slot | 有 | 换成不同E内容 | 内容干预，不是确认负证据监督 |

T_QT/T_PATH同架构各一个head，独立权重/optimizer，不线上集成。对照训练以G中的全部正确target为正，包括implicit，不用D替代G。

## Student

| 终点 | 学习参数起点 | 第二跳 | 当前任务训练是否用E | 正式作用 |
|---|---|---|---|---|
| SUP-QE | fresh SUP-C1 | (Q,E)条件化 | 是 | 无KD对照 |
| KD-QE | fresh KD-C1 | (Q,E)条件化 | 是 | 主模型 |
| KD-EONLY | 同一个fresh KD-C1 | E+受限MLP，不读Q | 是 | Q条件输入移除 |
| KD-NATIVE | 同一个fresh KD-C1，C2更新P/R | 原E→T，无MLP | 是 | 旧式结构/配方对照 |
| QT-SUP | 新table PCA/identity，无多模态任务前缀 | 不存在 | 否；共享无标签PCA例外 | 独立Direct-only训练 |
| QT-KD | 同一新table初值 | 不存在 | 无E训练样本，但T_QT教师曾做五关系warm-up | 独立Direct-only+QT蒸馏 |

“只看Q和T”与“全训练谱系从未使用任何多模态信息”不同；本包不做第二种更强声明。公共Qwen/PCA是共享初始资源。

## 三层比较，不相互冒充

1. **同候选Teacher：**锁定KD-QE C100及slot，比较T_QT/T_PATH-f0/T_PATH-Real/E-swap；区别训练配方与输入干预。
2. **固定reranker候选：**所有两路Student各自own C100交给同一T_QT；再对同一Student比较D100+T_QT与C100+T_QT。前者评价检索器，后者评价E带来的候选变化。
3. **独立完整系统：**QT-SUP/QT-KD own D100+T_QT、KD-NATIVE own两路C100+T_QT、主KD-QE own C100+T_PATH。比较完整配方，不宣称单因素因果。

Raw Qwen、历史B13/强Student/T0只在对应允许范围出现。历史参数/列表不得反流训练。

## 预算

原7阶段 + T_QT 1阶段 + KD-NATIVE-C2 1阶段 + 两个QT Student各C1/C2共4阶段 = 每seed13阶段，最多26。新增的仅是上表明确训练分支；同权重读出不重复计算成新模型训练。
