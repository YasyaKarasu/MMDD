# CLEAN-R1 采样等价提速补充协议

日期：2026-09-18。范围：CPU 候选列表构造；不修改模型、输入、训练目标或训练日程。

## 0. 结论与来源边界

本补充协议优先采用 EXACT 路线：同 corpus、GT、namespace、hard ranking 输入下，输出候选 ID、顺序、标签、mask 必须逐项等于原实现。它不是“分布近似相同”的采样替换。

已检查原实验文档第3、3.1、7.2、7.3节，以及原包 reference_core.py 中 stable_order。用户转贴的 DeepSeek profile 未在本环境独立复算，也没有本次服务器实际修改后的 MakeList 源码。因此下列代码是参考实现及集成合同，不是已经应用到服务器的 patch。

原规则是 SHA256((namespace + '\0' + id).encode('utf-8'))，按完整 digest、canonical ID 排序。用户要求的不改变训练效果，以保持训练输入流及计算过程不变为执行标准；不是承诺任何硬件环境都能逐位复现。

## 1. 不变量

保留全部原始 namespace 字段，包括 query、epoch、phase、packet、anchor、purpose。SUP/KD 的 sampling_arm 仍为 S，不新增 SUP/KD 盐。Teacher 和 Student phase 不合并；不同 query/epoch 绝不共用一个结果。

不更改：所有正例、Excluded、非法 ID 规则、hard 排名/刷新时机、16个hard与总31个负竞争项、最终小列表hash重排、query/packet/forward-chunk顺序、loss reduction、梯度累积、optimizer/scheduler步数、dropout/RNG状态、输入token/行/evidence数量、精度和模型初始化。继续遵守此前 Query 行保留修正，不回退到固定7组或前12行。

不要以“Teacher 无位置编码”为由删除候选顺序：本补丁直接保留原顺序，不靠浮点交换律或dropout分布等价证明安全。

## 2. 三项优先改动

### 2.1 单调用：完整哈希不变，去掉重复编码与全排序

每个 corpus 初始化一次 canonical ID 集合、UTF-8 bytes、ID映射；每个 namespace 初始化一次 SHA256((namespace+'\0').encode())，对每个ID复制其状态、update(ID bytes)、取完整32字节digest。

使用 heapq.nsmallest(k, (full_digest, canonical_id_rank) generator)。canonical_id_rank必须由UTF-8序建立。比较完整256bit，不把digest转float，也不只保留前64bit；并列时按原ID序。

这仍然扫描每个新namespace的合法corpus，不是从一个预筛子池抽取。效果等价不等于完全消除首次扫描成本。

原版 .hexdigest() 与等長 .digest() 的字典序一致；hash.copy()+update与哈希相同拼接字节一致。不要写成 SHA256(SHA256(namespace)+ID)，后者不等价。

原 deepcopy/global set/membership重复构造也可移出热循环；所有过滤规则须保持不变。

### 2.2 前缀缓存：31个ID覆盖hard变化，避免两Student重复全池哈希

令 C 是本packet的全部合法destination corpus；F为与hard选择独立的固定排除集合（P、Excluded、其他固定非法项），A=C\F。由HardRank取前16个合法不同ID构成H，h=|H|<=16。

本packet须从 A\H 按 uniform namespace 取 r=31-h 个ID；不足则全部取。先保存

    R = SHA_order(A, namespace)[:min(31, |A|)]

然后

    U = [x for x in R if x not in H][:31-h]
    negatives = H + U

为什么严格等价：从前31个元素最多删除h个，至少留下31-h个；因此A\H的前31-h个元素都位于R内。|A|<31时R已经是全部A。这个论证包含hard不足16的回填，而不只是正常15个uniform。

适用前提：补齐hard使用同一uniform hash顺序。如果服务器实现给hard-fill另设purpose/namespace，不能擅自改成上述顺序；应分别调用exact_topk，按其真实删除集合采用 k+删除上界 的前缀，或者保留原分支。必须先做差分测试确认。

缓存键必须包含：sampler算法/字节规则版本、corpus fingerprint、完整namespace、F的规范表示或fingerprint、negative_budget。H不进入前缀cache key（但在最终packet manifest中记录）。F发生变化不可复用旧前缀。完整packet缓存另需hard快照/最终H指纹。

SUP/KD可共享满足上述键的前缀，但必须分别按各自本epoch hard ranking得到H后解析，不可共享最终negative list。Teacher的phase不同，不共用该键。

不要跨epoch复用旧prefix。可提前构造本轮计划中各epoch各query的prefix，这不是复用历史train list，因为这些是当前GT/corpus/namespace推导的纯随机顺序前缀。

采用31个uint32 ID索引保存时，12630 queries ×6 epochs ×31 ×4=9,396,720 bytes，约8.96 MiB：这是一个packet、一组namespace的ID负载，不是包含元数据的总缓存；不要缓存每query的全corpus顺序或全部hash。

### 2.3 将CPU准备与GPU训练重叠，但不改变训练输入次序

保留训练主进程及原DataLoader随机行为，新增纯CPU采样准备池。初始以4个可用CPU进程、最多16个pending任务为性能工作点；不抢占用户未分配的CPU；不足4核则减少worker。worker数量是资源参数，不是训练超参数。

worker仅导入纯采样代码，不加载Qwen、Teacher、Student，不创建CUDA tensor；静态corpus通过initializer每个worker接收一次。任务只传namespace与小的排除列表，不能每query pickle完整21万对象。

使用spawn，且训练/模型初始化位于main guard内部，避免子进程导入时重建模型。主进程严格按原query/packet编号等待并消费结果；允许乱序计算，不允许按完成顺序训练。没有准备好时等待，不能跳过样本或换一个现成样本。

epoch hard刷新严格保留。可以提前算不依赖模型的prefix；完整packet必须使用该epoch实际对应的hard快照。预取不得把上个epoch的hard拿到下个epoch。

CPU prefetch不调用random/np.random/torch RNG；不以worker ID做随机盐。不因添加进程改变主进程global RNG与初始化顺序，须用测试确认。

## 3. 参考代码的接口

exact_sampling.py：
- Corpus.build：一次建立canonical ID/bytes索引；
- exact_topk：原stable_order(... )[:k]的精确替代；
- build_prefix / resolve_negatives：31个前缀与不同hard列表解析；
- ordered_prefixes：spawn、有限预取、输入序交付。

建议先接exact_topk做实际输出差分，随后接prefix复用与CPU预取。暂不改GPU前向chunk=8等计算顺序。若服务器现有NumPy top-k更快而且已验证完全等价，可以保留该核，只接prefix与预取；不要求为了形式一致换成heap。

缓存落盘可按phase/epoch/packet写分片manifest或二进制数组，不为每个query创建大量小文件；临时文件原子rename，两个进程写同键时需锁/单写者，防止读到半写数据。只保存成功完整前缀；出错按原规则重新计算，不随机降级。

## 4. 接入前后的验收

A. 使用原MakeList输出作为oracle。复用现有200-query基准，覆盖D/E/C、epoch1/3/6、正例/Excluded、不同hard快照。比较候选ID及顺序、P/N/unknown mask、label provenance、witness anchor、B、skip原因，必须100%一致。

B. 补充边界：hard=0…16、hard含重复/非法/正例、合法池<31、k=0、空池、Unicode ID、人工构造hash前64bit并列而全hash不同、全hash并列时ID tie-break。注意人工碰撞测试验证比较器，不需要制造真实SHA碰撞。

C. 比较串行与2/4个CPU worker输出；强制乱序完成时仍按原序交付。不同phase、epoch、query不能误命中缓存；SUP/KD只在合法键相同时共享prefix。

D. 从相同模型/optimizer/scheduler与Python/NumPy/torch CPU/CUDA RNG状态，使用相同设备和精度，重放20个optimizer updates，比较inputs、loss、梯度、参数。如果原实现自身同环境能bitwise复现，新实现也须bitwise相等；若原实现存在GPU非确定性，先做原版对原版复现对照并报告边界，不能把任何差异都自动认定为浮点噪声。

E. 保留原训练回执，记录implementation patch hash。精确等价通过后可从已有合法checkpoint继续，不要求因CPU优化重训；恢复仍使用原run RNG/step/epoch/corpus/GT。

## 5. 性能验证与本地结果的边界

拆分真实采样调用次数，而非用query次数推测：E的P为空时原协议会skip；Teacher epoch1/2无B且B原本复用D列表；阶段、重复调试及fresh-cache/cache-hit要分别计数。不要在B里重新调用大规模MakeList。

报告prefix冷构造、缓存命中解析、CPU队列等待、H2D、GPU forward/backward/optimizer、hard-mining、evaluation各自时间和端到端wall time。GPU操作异步，使用CUDA events或在计时窗口边界正确同步；不要每层同步破坏本要测的重叠。

用户转贴的289+26+7.6+90=412.6 ms，大于所称350ms/query，可能是测量窗口/重叠不同，不据此算精确加速倍数。

附带benchmark_exact_sampling.py使用211349个合成ID而非真实GT。BENCHMARK.json记录本环境每种实现3次、取中位数；Corpus构建单独记录。所有前缀ID与顺序相同，但没有与DeepSeek目前已优化的NumPy版本对比，也没有GPU训练计时。不得把本地算子加速比当作服务器或全流程加速比。

## 6. 其他优化的优先级

两个Student在Teacher完全冻结后彼此没有权重依赖。若实际有两张可用且各自能容纳对应整条训练臂的GPU，可以分别绑定设备运行；两臂hard/optimizer/output写路径必须隔离，prefix共享只读/安全缓存。每进程占满一张GPU不等于两张GPU不能各跑一臂；但若单臂实际占两卡、S-KD显存不足、其中一张不可用，则不成立。先修CPU再看并行，不自动启用DDP或合并两臂梯度。

检查训练进程是否误驻留冻结8B：原方案要求训练只读已缓存表示，此时不需要Qwen常驻GPU。仅凭16GB不能断言已误加载；需参数清单、allocated/reserved峰值和设备映射证据。

## 7. 本次不采用的替换

可以把 SHA256(namespace) 当局部PRNG种子，从全合法池无放回抽样；数学上保持均匀无放回目标分布，不必改成分层或移除query。但同seed抽出的具体ID与原hash排序不同，最终训练轨迹/指标不保证一致。因此不纳入此次EXACT补丁，不在已开始的主轨迹中切换，不声称“效果必然完全不变”。

不要改负例数、epoch、query行输入、温度、loss、hard刷新、模型维度、精度，来解决CPU列表构造问题。

## 8. 核查来源

- 原包 EXPERIMENT_SPEC.zh-CN.md 第3/3.1/7.2/7.3节；reference_core.py 中 stable_order。
- Python hashlib（共同前缀copy、update拼接等价、digest、短消息GIL边界）：https://docs.python.org/3/library/hashlib.html
- Python heapq（nsmallest与完整排序前k等价）：https://docs.python.org/3/library/heapq.html
- Python concurrent.futures（ProcessPoolExecutor）：https://docs.python.org/3/library/concurrent.futures.html
- NumPy lexsort（是完整间接稳定排序，不是自动的partial selection）：https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
- PyTorch 2.9 CUDA计时与可复现性：https://docs.pytorch.org/docs/2.9/notes/cuda.html ; https://docs.pytorch.org/docs/2.9/notes/randomness.html

模型相关参数和原始实验协议以本轮用户批准修正后的版本为准。本文件只扩充等价执行方式，不重置实验语义。


---

## 9. 集成回执（2026-09-18 由 DeepSeek Harness 在本仓库完成，非原包作者）

第 0 节说"下列代码是参考实现及集成合同，不是已经应用到服务器的 patch"。本节记录它**已经
被集成并做过哪些等价验证**，以及哪些仍待有卡机器确认。详细回执见
`work/s1_cpu_speedup_r1/INVARIANTS.zh-CN.md`。

**已集成**：`src/mmdd_stage1_clean/sampling.py`（单次扫描取 hash 前 31、前缀缓存、
`build_legacy` oracle）、`src/mmdd_stage1_clean/util.py`（`HashPrefixStore`、`hash_prefix_key`、
`cached_hash_prefix`）、`src/mmdd_stage1_clean/train.py`（E 短路 gate、B 复用 D 对象、
`packet_queue_wait`/`step` 分离计时、学生侧 `want`）、`commands.py` + `__main__.py`
（`--prefix-cache`、学生 `--prefetch-workers` 真正接线）、
`src/tests/test_clean_r1_cpu_speedup.py`（46 项边界/缓存隔离/gate 测试）。

**已验证（CPU）**：与改动前 `sampling.py` 的**逐字节冻结副本**差分 200 query × epoch {1,3,6}
× {teacher,student} = 1200 次比对 / 2716 个 packet，候选 ID 与顺序、整表顺序、正负例、
P/N/unknown 标签、provenance、witness anchor、B context/B 正例覆盖、skip 分类 **0 差异**；
核心算子穷举 144 项 0 差异；workers 1/2/4 逐包相同且全局 RNG 未变；20 个 optimizer update
在 workers=1/4 下 loss/梯度/参数/RNG 逐位相同；冷 75.3ms→热 5.98ms per build（x12.7），
真实两臂场景 x3.45。

**未验证（必须在有卡机器）**：CUDA 上的 20-step replay（并要求先做"原版对原版"复现）、
S-SUP/S-KD 峰值显存与 device mapping、CUDA 异步计时边界同步。器械：
`work/s1_cpu_speedup_r1/replay_20_step.py`、`gpu_profile.py`。

第 2.2 节"首先核查 hard-fill 是否与 uniform 同 namespace"的结论：**同 namespace**，
`build` 的 fill 与 uniform 都传 `PURPOSE_RANDOM`（v2/v3 版本一致），因此前缀合并安全。
第 2.1 节"若当前核更快且相同，则保留当前核"：**保留** `heapq.nsmallest` 核，实测 k=31 与
k=62 同耗时，说明已是部分选择而非全排序；本次加速来自缓存与消除重复扫描。
