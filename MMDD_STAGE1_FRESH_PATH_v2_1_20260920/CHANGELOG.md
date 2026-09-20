# v2.1变更说明

相对v2.0，只为当前用户要求新增对照并修订硬件/调度；原主线架构、64摘要合同、PCA、raw采样、SUP/KD C1、三条条件化C2及其损失/epoch保持。

1. 新增独立T_QT：同run同seed T_EDGE分叉、2epoch、QT目标排名+五关系pair replay；不是T_PATH-f0。
2. 新增QT-SUP/QT-KD：新table P/R，1epoch C1+3epoch全目标QT C2，严格Direct-only线上。
3. 新增KD-NATIVE：从本run KD-C1开始，1epoch P/R无条件ET结构C2；明确其新配方与历史B13不等价。原KD-EONLY有MLP，不删不改名。
4. 同池Teacher、固定reranker候选、独立完整系统三张比较表及历史兼容性清单；原Raw与implicit/explicit/strict/own-pool要求保留。
5. 硬件仅2×RTX4090；授权双卡独立作业与profile通过后的同卡2作业，保持科学语义与缓存/RNG隔离。
6. 预算7→13阶段/seed、上限14→26；repeat gate增加对独立T_QT的-0.005容忍界，不削弱原其余门槛。

本版未声称上述对照已训练或能够达到历史分数。配方系预先指定工作点，不是已验证最优。
