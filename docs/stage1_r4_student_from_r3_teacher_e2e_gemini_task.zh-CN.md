# Stage-1 R4 → Stage-2：从 R3 Teacher 蒸馏学生、学生池增强 Teacher、端到端评测（EntiTables，seed 13，双卡）

给执行者（Gemini）：本文分四部分，**按顺序执行**，每部分完成并通过检查再进入下一部分。
本轮的主角是"把已经修好的 R3 Teacher 真正接进学生训练链"，然后把它一路做到端到端 recall/nDCG。

背景文档：
- R3 结论与数据：`docs/stage1_r3_teacher_evidence_residual_report.zh-CN.md`
- R3 任务说明：`docs/stage1_r3_teacher_evidence_residual_task.zh-CN.md`
- 上一轮续训说明（同款脚本）：`docs/stage1_teacher_student_pool_continuation_gemini_task.zh-CN.md`
- 学生蒸馏诊断：`docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md`

---

## 0. 规则（必须遵守）

- 所有命令在仓库根目录 `/home/oycy/MMDD` 下执行，conda 环境 `MMDD`。时间一律 `date -u`。
- **运行期间不要修改 `src/`、`tests/` 下的任何文件**，除非第 2 节明确要求你做代码改动。改过源码后
  `validate`/`smoke`/`train` 会拒绝执行，需要按第 2.3 节末尾的流程处理。
- 不要读取、查看或修改 `.env.openai`。本实验不需要它。
- 不要删除、移动或修改 `work/stage1_features/`（多个 run 共享的冻结特征）。
- 不要修改任何已有 `work/stage1_r3_*`、`work/stage1_entitables_r2_fullkd_s13*` 目录；它们只读。
- 只跑本文这一个 run 链，不要同时跑其它 run。
- gate 判定为 FAIL 是研究结论，不是程序错误。照常完成并**如实报告**，不要为了通过而改配置重跑。
- `train`、`train-side`、续训、Stage-2 `features`/`recover` 必须 `setsid nohup ... &` 脱离会话，否则
  执行者回合结束会被杀掉。每次启动前先 `date -u` 记录，启动后把 pid 写入文件。

---

## 1. 这一轮要回答什么

| # | 问题 | 判定方式 |
|---|---|---|
| Q1 | 用修好的 R3 Teacher 蒸馏，学生的 E 池覆盖能否从被压垮的 0.475 回到 ≥0.85？ | `seed13/eval/dev/native_kd/METRICS.json` 的 `E_target_coverage` |
| Q2 | 学生自身的直接检索 R@10 是否变好（implicit 子集）？ | 同上 `Direct_ANN_R10`，与 R2 学生 0.5022 / 0.4225 对比 |
| Q3 | 从新学生池挖掘难例继续训 Teacher，Teacher 能否在新学生池上 `Real ≥ 学生 Direct`？ | 续训产物 `eval/dev/SUMMARY.json` 的 `end.Real` vs `student Direct_ANN_R10` |
| Q4 | 用新学生（student 召回 + Teacher 重排）导出的 handoff，训练新 Stage-2 selector 后，端到端 R/nDCG@5,10,15,20 是多少？ | Stage-2 `evaluation/METRICS.csv` |
| Q5 | 图片裁剪对端到端指标有没有可测贡献？ | 同 handoff 的 crop-on 与 crop-off 两臂对照（第 4.5 节） |

**注意**：Q1/Q2 是这一轮的科学核心；Q4 是"把故事做到端到端"的交付；Q5 是裁剪对照。

---

## 2. ★ 第一部分的门槛：把 R3 Teacher 接进学生链需要一次代码改动

### 2.1 现状（已核实，不是猜测）

学生 C2 的 KD 蒸馏，其 Teacher 只能来自本 run 自己的
`<run>/seed13/TB_CQET/checkpoints/end.pt`：

- `src/mmdd_stage1/pipeline.py:1379` — TB_CQET 由 `_tb_stage(...)` 在本 run 内训练产生；
- `src/mmdd_stage1/pipeline.py:1427` — `cqet_teacher = _load_teacher(tb_cqet)`，随后 `build_teacher_logits_cache` 用它打分；
- `src/mmdd_stage1/pipeline.py:1440-1442` — 该 checkpoint 的路径与 state hash 被写进 `teacher_logits_cache/identity.json`；
- `src/mmdd_stage1/export.py:123` — Stage-2 导出同样用 `<run>/seed13/TB_CQET/checkpoints/end.pt` 做重排。

而 R3 的 Teacher 在**两个 out-dir**里，都不是 run root：

| 目录 | 产物 | 是不是 run root |
|---|---|---|
| `work/stage1_r3_chain_residual_A/` | `TA/{init,epoch1,epoch2}.pt`、`TB_CQET/{init,half,end}.pt` | 否（无 `protocol.json`、无 `seed13/`、无 pipeline 收据） |
| `work/stage1_r3_teacher_sp_A/` | `checkpoints/{init,half,end}.pt`（学生池续训产物） | 否 |

`import-teacher --from-run` 要求源是**完成的 run root**，且 `teacher`/`retrieval`/`feature_provenance`
三个 block 与当前 run 完全一致（`pipeline.py:178-217`）；out-dir 既没有 `protocol.json`，`TA/TB_CQET`
也不带 pipeline 收据格式，因此**不能被 `import-teacher` 复用**。

结论：**没有任何现成命令能把 R3 的 Teacher 放到学生 KD 面前。**

### 2.2 代码改动（已实现，无需执行者编写）

子命令 `adopt-teacher` 已加入源码：把一个外部 Teacher 检查点**安装**成当前 run 的 `seed13/TA` 与
`seed13/TB_CQET` 两个 stage，并写出 `train` 能识别为"已完成"的 stage 收据。

- 实现：`src/mmdd_stage1/pipeline.py` 的 `adopt_teacher_chain(protocol_path, run_root, *, ta_dir,
  tb_dir, path_mode=None, reason, replace=False)`；CLI 在 `src/run_stage1.py`：
  `$S adopt-teacher --run-root $RUN --ta-dir DIR --tb-dir DIR [--path-mode triplet|pairwise_residual]
  [--replace] --reason TEXT`。测试：`tests/test_stage1_cqet.py::test_adopt_teacher_chain_installs_external_checkpoints`。
- `--ta-dir` 需含 `init.pt / epoch1.pt / epoch2.pt`，`--tb-dir` 需含 `init.pt / half.pt / end.pt`
  （即 `_completed_stage_result` 与 Teacher 轨迹读取的全部名字）。`$R3A/TA` 与 `$R3A/TB_CQET`
  正好是这两个目录。
- `--path-mode` 默认取协议的 `teacher.path_mode`，若与协议不一致会直接报错；它必须与检查点自身的
  架构一致（有 `path_head.*` 键即 `pairwise_residual`）。
- 语义与 `import-teacher` 一样要求 `lock` 之后、`prepare` 之前；每个复制的文件都会被重新 hash 并写进
  收据，来源写入 `<run>/ADOPTED_TEACHER_CHAIN.json`。
- **收据的 source identity 写的是当前源码**（`record_stage_pre_run` 的行为），所以本轮**不需要**
  `amend-source`；`_completed_stage_result` 的 source 检查会直接通过。
- **幂等**：`TB_CQET` 已存在时要再安装（例如 4.2 续训之后），加 `--replace`，会以新的
  `attempt` 覆盖旧的 `checkpoints` 链接。
- **TB_QT 不在安装范围**：R3 没有产出 TB_QT，`train-side` 会从安装的 TA 正常训练 TB_QT（约 25 min）。
  这正是我们要的。

### 2.3 跑之前先确认代码已就绪

1. 跑测试：`conda run -n MMDD python -m pytest tests/test_stage1_cqet.py -q`（应 63 passed）；
2. `lock` 之后立即 `adopt-teacher`，然后 `prepare`；确认 `train` 启动后**跳过了 TA 与 TB_CQET**
   （重跑 `train` 时这两个 stage 的 `POST_RUN` 已存在，`_run_stage` 会直接复用，不会重新训练）。

### 2.4 如果你不想改代码（fallback，明确更弱）

在 run 内用 R3 的配方（`path_mode=pairwise_residual`、`path_loss_scope=bagged`、
`witness_target_weight=0.5`、`support_weight=1.0`、`support_competitors=16`）从 init 重新训一个 Teacher，
再让学生蒸馏。代价：多烧 TA+TB_CQET 约 2.2 h，**而且得到的不是 R3 那个 Teacher**——in-run 的 TB_CQET
不会把 TA 的 `qet_lists` 并进 TB 记录，`witness_target_weight` 因此没有可作用的列表，效果介于 R3 的
对照臂 B 与主臂 A 之间。除非时间极紧，不要选这条。

---

## 3. 变量

```bash
cd /home/oycy/MMDD
R2=work/stage1_entitables_r2_fullkd_s13          # 数据集/特征/列表的来源 run，只读
R3A=work/stage1_r3_chain_residual_A              # R3 第一步：残差 + 新损失的主臂 Teacher
R3SP=work/stage1_r3_teacher_sp_A                 # R3 第二步：学生池续训 Teacher（部署产物）
RUN=work/stage1_entitables_r4_s13                # 本轮新 run（必须不存在）
SP=work/stage1_r4_teacher_sp                     # 本轮续训 Teacher
EXPORT=work/stage1_entitables_r4_s13/stage2_handoff
S2A=work/stage2_entitables_r4_crop_on            # Stage-2 crop-on 臂
S2B=work/stage2_entitables_r4_crop_off           # Stage-2 crop-off 臂
DATASET=output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9
GPU=0
SIDE=1
P=/home/oycy/miniconda3/envs/MMDD/bin/python
S="conda run --no-capture-output -n MMDD python src/run_stage1.py"
```

---

## 4. 执行步骤

### 4.1 第一部分：新 run + 安装 Teacher + 训学生

跑之前检查（全部满足才继续）：

```bash
cd /home/oycy/MMDD && date -u
nvidia-smi --query-gpu=index,name,memory.used --format=csv     # 两张 4090 都空闲
free -g                                                         # available ≥ 90 GiB，且无其它大内存任务
git status --short
ls work/stage1_features/entitables/features/z/z.f32.npy
ls $R3A/TB_CQET/end.pt $R3A/TA/epoch2.pt $R3SP/checkpoints/end.pt
```

Step 1 — 初始化（复用冻结特征，不跑 build-data / encode）：

```bash
$S init --run-root $RUN --dataset-root $DATASET \
  --features-dir work/stage1_features/entitables --gpu $GPU --side-gpu $SIDE --seeds 13
# 期望：打印 existing encoding found; skip build-data and encode；protocol.json 里 gpu_processes=2
```

Step 2 — 锁：

```bash
$S lock --run-root $RUN          # 约 5 min；完成标志 SOURCE_AND_CACHE_LOCKED_PENDING_NUMERIC_PROBE
```

Step 3 — **安装 R3 主臂 Teacher**（`adopt-teacher`，见第 2 节；`$R3A/TA` 与 `$R3A/TB_CQET` 已含所需名字）：

```bash
$S adopt-teacher --run-root $RUN \
  --ta-dir  $R3A/TA \
  --tb-dir  $R3A/TB_CQET \
  --path-mode pairwise_residual \
  --reason "R3 residual teacher (raw pool, main arm A) into the student KD path"
```

Step 4 — 特征与参考测试：

```bash
$S verify-features --run-root $RUN     # PASS_READY_FOR_REFERENCE_TESTS
$S prepare         --run-root $RUN     # PASS_READY_FOR_VALIDATION
$S validate        --run-root $RUN     # PASS_READY_FOR_SMOKE
```

Step 5 — 冒烟（必须后台）：

```bash
mkdir -p $RUN/logs
setsid nohup $S smoke --run-root $RUN > $RUN/logs/smoke.log 2>&1 < /dev/null &
# 轮询 $RUN/tests/smoke/LATEST.json，status 应为 PASS（PASS_READY_FOR_FORMAL）
```

**冒烟通过后立刻做一次关键核对**：确认 Teacher 是安装进去的、不是重训的。

```bash
sha256sum $R3A/TB_CQET/end.pt
cat $RUN/seed13/teacher_logits_cache/identity.json | head -20
# identity 里的 teacher_checkpoint / teacher_checkpoint_sha256 必须指向安装进去的那个文件
```

Step 6 — 正式训练，双进程后台：

```bash
setsid nohup $S train      --run-root $RUN > $RUN/logs/train.log      2>&1 < /dev/null & echo $! > $RUN/logs/train.pid
setsid nohup $S train-side --run-root $RUN > $RUN/logs/train_side.log 2>&1 < /dev/null & echo $! > $RUN/logs/side.pid
```

- `train`（主 GPU）：跳过 TA/TB_CQET（已安装）→ Native C1 → C2 共享图 → Native C2 SUP/KD → dev 评估；
- `train-side`（第二 GPU）：TB_QT（从安装的 TA 训练）→ QT C1/C2 → Teacher 轨迹 → test 评估。

轮询 `$RUN/PHASE_STATUS.json` 与 `$RUN/processes/{main,side}.json`；两者都 `COMPLETE*` 才算结束。
**预计 wall time 约 2–3 h**（省掉了 R2 那 76 min TA + 58 min TB_CQET，剩 TB_QT 25 min + 学生 + 评估；
若安装校验失败回退到重训 Teacher，则约 5–6 h）。

**Q1/Q2 的读法**（完成后立即汇报，不要下结论）：

```bash
python3 -c "
import json
d=json.load(open('work/stage1_entitables_r4_s13/seed13/eval/dev/native_kd/METRICS.json'))
for seg in ('overall','implicit','explicit'):
    x=d['candidate'][seg]
    print(seg, 'Direct_ANN_R10=%.4f'%x['Direct_ANN_R10'], 'E_target_coverage=%.4f'%x['E_target_coverage'])
"
```
对照：R2 KD 学生 `Direct_ANN_R10` 0.5022 / implicit 0.4225；E 池覆盖 0.4749。

Step 7 — **可选但推荐**：把第二起点（`$R3SP/checkpoints/end.pt`）也跑一遍（另一个 run 根 + 另一次
`adopt-teacher`），比较哪个 Teacher 起点给出更好的学生 E 覆盖与 implicit R@10。两个 run 不要同时跑。

### 4.2 第二部分：从新学生池挖掘难例，续训 Teacher

入口与 R3 完全一致，**起点换成新 run 自己的学生与 Teacher**：

```bash
$P src/continue_teacher_on_student_pool.py --run-root $RUN --out-dir $SP \
  --gpu $SIDE --student KD \
  --init-checkpoint $RUN/seed13/TB_CQET/checkpoints/end.pt \
  --path-mode pairwise_residual --path-loss-scope bagged \
  --witness-target-weight 0.5 --support-weight 1.0 --support-competitors 16 \
  --search exact > $SP.log 2>&1 &
```

- 先 `--limit 16` 冒烟（约 10 min），确认 mine/train/evaluate 三段都通，再跑正式；
- 预计：挖掘 ~20 min + 训练 ~2.2 h + 评估 ~20 min；
- 完成标志：`$SP/eval/dev/SUMMARY.json` 存在，且 `checkpoints/{init,half,end}.pt` 三个都在。

**Q3 的读法**：`end.Real`（native_kd 池）应 ≥ 新学生的 `Direct_ANN_R10`，且 ≥ `init.Real`。
同时报告 implicit 子集的 `Real − Swap` 点估计与（若脚本给出）配对 bootstrap 区间。

### 4.3 第三部分：导出 handoff（student 召回 + Teacher 重排）

导出的语义（`src/mmdd_stage1/export.py`）：对该学生的 C150 池做 **TB_CQET end 重排**，写
`retrieval.{train,dev,test}.jsonl`。其中：

- **train** 由 `_materialize_train` 现算（学生检索 + `TB_CQET` 重排）；
- **dev/test 不重算**，直接复用 run 冻结评估目录 `seed<seed>/eval/{dev,test}/native_kd/`
  里的 `rankings.TB_CQET.Real.jsonl.gz` 与 `logits.TB_CQET.Real.jsonl.gz`（`export.py:96-110`）。

**必须注意的事实**：dev/test 用的 Teacher 是**第一部分安装进 run 的那个**（R3 主臂），
不是 4.2 里续训出的 `$SP`。也就是说 **Q4（端到端指标）反映的是 R3 主臂 Teacher，而不是续训后的
Teacher**；`$SP` 的收益只在 Q3（Stage-1 Teacher 层）体现。这是刻意的：R3 报告已表明主臂 Teacher
在 raw 池上 Real−Swap 显著，续训教师只在小得多的学生池上再训 1 epoch，把它塞进 dev/test 会让
两轮不可比。**不要**为了"用上续训教师"而手改 eval 目录。

导出命令：

```bash
$S export --run-root $RUN --output-dir $EXPORT --arm KD --top-k 50 --evidence-path-k 4
# 产物：$EXPORT/retrieval.{train,dev,test}.jsonl、$EXPORT/stage1_gate.json
```

导出完成后核对两点（写进汇报）：

```bash
python3 -c "
import json
g=json.load(open('work/stage1_entitables_r4_s13/stage2_handoff/stage1_gate.json'))
print('completed_stage:', g.get('completed_stage'))
print('selection_split:', g.get('selection_split'))
sel=json.load(open('work/stage1_entitables_r4_s13/seed13/selections/NATIVE_C2_COMMON.json'))
print('KD_checkpoint matches:', g.get('student_checkpoint')==sel.get('KD_checkpoint'))
"
wc -l $EXPORT/retrieval.train.jsonl $EXPORT/retrieval.dev.jsonl $EXPORT/retrieval.test.jsonl
```

**仅当时间充裕**才做这个额外变体（不进主线）：在把 `$SP` 装入 `TB_CQET` 后，让 `evaluate_split`
用新 Teacher 重算 dev/test（换新 run 根重跑评估段，或按报错提示处理），这样 dev/test 才反映续训教师。
它会让本轮与 R3 的报告口径不一致，必须在汇报里写明。

### 4.4 第四部分：训练新 Stage-2 selector

```bash
S2="conda run --no-capture-output -n MMDD python src/run_stage2.py"
$S2 init --run-root $S2A --stage1-handoff $EXPORT
$S2 catalog --run-root $S2A                 # CPU
$S2 jobs    --run-root $S2A                 # CPU
$S2 features --run-root $S2A --split train --gpu $SIDE   # GPU
$S2 features --run-root $S2A --split dev   --gpu $SIDE
$S2 features --run-root $S2A --split test  --gpu $SIDE
$S2 train-head --run-root $S2A              # CPU，20 epoch
$S2 plans      --run-root $S2A              # CPU
$S2 recover    --run-root $S2A --gpu $SIDE  # GPU，9B 行级恢复，最耗时
$S2 score      --run-root $S2A              # CPU
$S2 evaluate   --run-root $S2A              # CPU
```

- `init` 会把模板 `configs/mmdd_stage2_bidf.json` 写成 `$S2A/config.json`，**只能写一次**；
- `features`/`recover`/`score` 可断点续跑（已有文件会跳过）；
- 预计：`recover` 约 3–6 h（9B，2364 query，据 E2E run 平均约 4.7 s/query 的 warm pipeline 外加生成墙钟）；
  `features` 每 split 数十分钟到数小时；其余 CPU 步合计 <1 h。
- 4.5 的 crop-off 臂与 crop-on 臂**串行**执行，不要并行抢同一张卡。

### 4.5 图片裁剪对照（Q5）

同一份 handoff、同一套配置，只改裁剪策略，跑第二个 run 根。`config.json` 不可重入，必须 `init` 新目录后
**在跑 `recover` 之前**改配置。

```bash
$S2 init --run-root $S2B --stage1-handoff $EXPORT
python3 -c "
import json
p='work/stage2_entitables_r4_crop_off/config.json'
d=json.load(open(p))
d['crop']['min_joint_concentration']=2.0   # ≥1 → 任何 heatmap 都判 DIFFUSE → 全部 fallback，不产生 crop
json.dump(d,open(p,'w'),indent=2,ensure_ascii=False)
print('crop-off arm configured')
"
# 然后对 $S2B 重复 4.4 的 catalog→…→evaluate
```

- crop-off 臂里生成器**只看到原图**，与 crop-on 臂的唯一差别就是有没有 context/tight 裁剪视图；
- **注意**：crop-off 臂仍会跑一次定位前向（只是拒绝所有 ROI），所以这是**内容对照**，不是算力对照；
- 汇报 `$S2A/evaluation/METRICS.csv` 与 `$S2B/evaluation/METRICS.csv` 的 **BIDF_RRF60** 行，
  overall/implicit/explicit 三个 population、R 与 NDCG 的 5/10/15/20 四个 k，以及两臂 `CONTRASTS.csv`。
  裁剪是否留下，取决于这个差值的点估计与区间，**不要预设它有用**。

---

## 5. 指标口径（Q4/Q5）

- Stage-2 `evaluation/METRICS.csv` 的 cutoffs 来自配置，默认 `[1,5,10,15,20,30,50]`，
  **k=5,10,15,20 都在里面**，无需改代码。策略行取 `STAGE1`（一阶段基线）、
  `BIDF_RRF60`（本方法）、`VISIBLE_IDF_RRF60`（不看证据的对照）。
- Stage-1 层的 recall 只在 `(10,20,30,40,50)` 有输出（`src/mmdd_stage1/metrics.py:23`，硬编码），
  **k=5 与 15 在 Stage-1 层不可得**；要 k={5,10,15,20} 的统一口径，用 Stage-2 的 `METRICS.csv`。
- 汇报时同时给 **dev 与 test**，并明确 test 只是"冻结后的历史暴露回归"（协议 `evaluation.test_role`），
  不是泛化证据。

---

## 6. 这个任务不做的事

- 不改 Stage-1 的 `path_mode`/损失/协议模板（除非走 2.4 的 fallback）；
- 不改 Stage-2 的 `candidate_scope`、`rrf_constant`、`matching.tau_cosine` 或恢复输出上限；
- 不用 query 标签（implicit/explicit）做路由或选样；
- 不为了让指标好看而挑选样例、调阈值或补选数据；
- 不在同一张 GPU 上并行两个 9B 恢复任务。

---

## 7. 汇报模板（每部分完成后）

```
[部分 N] 状态：COMPLETE / FAILED / PARTIAL
命令：(原样)
时间：(UTC 起止，wall)
产物：路径 + 关键文件 sha256（若有）
关键数字：(只列原始数值，不下结论)
异常：(报错原文、日志路径；没有就写 none)
```
