# 本次只做逐样本完全一致的CPU采样提速

请不要改模型、GT、query行数、evidence、负例数、loss、温度、batch/forward chunk、训练步数、hard refresh或candidate消费顺序。不要把SHA256换成PRNG，不删除query/epoch/anchor namespace。无需因为本次等价优化从头重训。

请读 OPTIMIZATION_NOTE.zh-CN.md，先以当前真正使用的MakeList/stable_order_topk为oracle，保留其实现供差分测试，再执行：

1. 确认numpy lexsort作用在哪个规模上：全N上lexsort再[:k]仍是全排序。优先实现完整SHA256 bytes + 原ID tie-break的exact top-k；使用预编码ID和SHA256(namespace+'\0')的copy/update减少重复工作。可借鉴exact_sampling.py；若当前核更快且相同，则保留当前核。
2. 对每个完整namespace和固定排除集合，缓存合法基础池hash前31个ID；16个hard过滤后取31-|H|个，包含hard不足回填。SUP/KD共享的是前缀，不能共享两臂不同hard产生的最终训练列表。首先核查hard-fill是否与uniform同namespace；若不一致，不强行合并。
3. 增加纯CPU进程预取，默认4个可用CPU进程、最多16任务；保留GPU主进程训练顺序与RNG。worker不import/加载Qwen和训练模型，不运行CUDA，不传全corpus大对象每任务。epoch hard snapshot不可串用。
4. B packet直接复用D packet的target list；本应skip的E/C不做无用hash。与原协议相同的skip处理不能增删训练样本。
5. 用现有200-query基准比对修改前后所有候选ID、顺序、labels/masks、witness/B、skip；补齐本包边界测试；从相同训练状态重放20个optimizer updates，核对RNG/损失/梯度/权重。全部通过后再接回正式运行。GPU已有不确定性需先原版对原版核查，不能随意容忍新差异。
6. 重新报告冷构造、cache-hit、队列等待、GPU前反向、hard mining与端到端耗时，正确处理CUDA异步计时。不复用未经确认的70%/82%或25–30小时外推。
7. 在CPU瓶颈修复后检查实际两卡device mapping与S-SUP/S-KD峰值显存；如果确实两卡各能完整容纳对应一臂，Teacher冻结后可以各卡并行。不自动开启DDP、不共享训练权重、不覆盖彼此hard/output文件。

交付：实际修改diff、协议不变量检查、真实差分与20-step回执、profile对比、运行命令与代码hash。包内14项CPU测试和合成benchmark不代表服务器集成已成功，不能拿它们替代实际验收。
