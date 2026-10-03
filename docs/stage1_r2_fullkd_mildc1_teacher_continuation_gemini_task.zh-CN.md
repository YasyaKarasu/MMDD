# Stage-1 第二轮：全列表 KD + 温和 C1 + Teacher 学生池续训（EntiTables，seed 13，双卡）

给执行者（Gemini）：按顺序执行，每一步检查完成标志再进入下一步。本文是两段任务：
**第一部分**跑一个新的 Stage-1 正式 run；**第二部分**在它结束后，用它选出的 KD 学生做 Teacher 续训（主臂 + 对照臂）。
背景见 `docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md`（诊断）、`docs/stage1_plan_abcd_run_guide.zh-CN.md`
（上一轮 run 的说明）、`docs/stage1_teacher_student_pool_continuation_gemini_task.zh-CN.md`（上一轮续训的说明）。
实现说明见 `src/README.md` 的 "Stage-1 main flow" 一节。

## 0. 这一轮改了什么，为什么

上一轮 run `work/stage1_entitables_abcd_s13`（方案 A–D 全开）跑完了，三个 gate 全部 FAIL。事后归因（direct-only 探针，
`work/stage1_entitables_abcd_s13_probe/`）和 Teacher 续训（`work/stage1_entitables_abcd_s13_teacher_sp{,_raw2}`）给出三个结论，
对应本轮三处改动。**代码与协议模板已经改好，执行者不需要改任何文件。**

| 改动 | 协议键 | 上一轮 | 本轮 | 依据 |
|---|---|---|---|---|
| KD 用全列表 KL，关掉 top-50 + rank-mass 尾项 | `student.kd_top_k` | 50 | **0** | 其它条件相同，只把 KD 从全列表 KL 换成 top-50，dev Direct R@10 掉 4–5pp，正是上一轮 KD < SUP 的全部幅度。全列表 KL 下 SUP+KD 0.470 vs SUP 0.436（dev，2/3 处） |
| C1 用单独的、更温和的学习率 | `student.C1.P_lr` / `student.C1.R_lr` | 继承 1e-4 / 1e-3 | **1e-05 / 0.0001**（依据见第 7.1 节） | 共享学习率下 C1 把 dev E 池覆盖从 0.68 打到 0.33，C150 优先的选点退回 fraction 0（identity），C1 等于白跑 |
| 跑完后接一轮 Teacher 续训 | 不在协议里，用 `src/continue_teacher_on_student_pool.py` | 已跑过一次 | 在新 run 上重跑主臂 + 对照臂 | 上一轮：学生池列表续训 1 epoch 后，Teacher 在 KD 学生 dev 池上 Real R@10 0.394 → 0.488（+9.3pp，CI [7.0, 11.6]），高出学生自己的直接检索 6.0pp；对照臂（同起点、Raw 列表再训 1 epoch）−0.3pp |

其它开关不变：`teacher_scored_negatives=true`、`lr_schedule=cosine`、`evidence_random_negatives=256`（方案 D，仍未消融）、
C2 3 个 epoch、τ=10、`kd_weight=1`。`evidence_random_negatives` 故意保留：上一轮 KD 臂的 E 池覆盖比 SUP 低 10.6pp（CI 不跨 0），
top-50 关掉之后如果这个差距还在，就能归到方案 D 的 evidence KD 上。

上一轮正式 run 的参照数字（dev，seed 13）：

| 量 | 上一轮 |
|---|---:|
| C1 选中 fraction | 0（identity） |
| C1 fraction 0 / 1.0 的 E 池覆盖 | 0.683 / 0.333 |
| C2 SUP / KD 学生 Direct_ANN_R10（2/3 处，选中点） | 0.4436 / 0.4279 |
| C2 SUP / KD 的 E 池覆盖 | 0.852 / 0.746 |
| TB_CQET f0 在 Raw 池 / SUP 池 / KD 池 | 0.452 / 0.410 / 0.387 |
| 续训后 Teacher 在 KD 池上 Real（init → end） | 0.394 → 0.488 |

## 1. 规则（必须遵守）

- 所有命令在仓库根目录 `/home/oycy/MMDD` 下执行，conda 环境 `MMDD`。时间一律 `date -u`（UTC）。
- **运行期间不要修改 `src/`、`tests/`、`configs/` 下的任何文件。** `validate`、`smoke`、`train`、`train-side` 绑定源码哈希，
  改了源码后续步骤会拒绝执行。遇到代码报错：停下来报告报错与日志路径，不要自行打补丁，不要 `amend-source`。
- 不要读取、查看或修改 `.env.openai`。本实验不需要它。
- 不要删除或覆盖 `work/stage1_features/` 下的任何内容（多个 run 共享的冻结特征，只读）；也不要碰上一轮的
  `work/stage1_entitables_abcd_s13*` 目录。
- 只跑本文这一个 run，不要同时跑其它 run。
- gate 判定为 FAIL 是研究结论，不是程序错误。照常完成并如实报告，不要为了让 gate 通过而改配置重跑。
- 本机首次 CUDA 初始化偶尔报 `CUDA driver initialization failed`（`cuInit` 返回 3）。不是权限问题，原样重跑一次该命令即可。
- `smoke`、`train`、`train-side`、续训 都必须 `setsid nohup ... &` 脱离会话在后台运行，否则你的回合结束时进程会被杀掉。
- 定期轮询（10–15 分钟一次），不要原地阻塞。

## 2. 运行前检查

```bash
cd /home/oycy/MMDD
git status --short            # 记录进最终报告。预期：configs/mmdd_stage1_cqet_protocol.json、src/mmdd_stage1/{config,pipeline,train}.py、
                              # src/README.md、tests/test_stage1_cqet.py 为已修改或已提交；另有续训脚本及其文档/测试
git log --oneline -1
python3 -c "
import json; s=json.load(open('configs/mmdd_stage1_cqet_protocol.json'))['student']
print('kd_top_k', s['kd_top_k']); print('C1 lr', s['C1'].get('P_lr'), s['C1'].get('R_lr')); print('shared lr', s['P_lr'], s['R_lr'])
print('teacher_scored_negatives', s['teacher_scored_negatives'], 'lr_schedule', s['lr_schedule'], 'evidence_random_negatives', s['evidence_random_negatives'], 'C2 epochs', s['C2']['epochs'])"
nvidia-smi --query-gpu=index,name,memory.used --format=csv     # 两张 4090 都应空闲（memory.used 接近 0）
free -g                                                         # available 必须 ≥ 90 GiB；机器上不要有其它大内存任务
ls work/stage1_features/entitables/features/z/z.f32.npy         # 冻结特征必须存在
ls -d work/stage1_entitables_r2_fullkd_s13 2>/dev/null && echo "RUN DIR ALREADY EXISTS - STOP AND REPORT"
```

协议打印必须是：`kd_top_k 0`、`C1 lr 1e-05 0.0001`、`shared lr 0.0001 0.001`、`teacher_scored_negatives True`、
`lr_schedule cosine`、`evidence_random_negatives 256`、`C2 epochs 3`。不符就停下来报告，不要自己改。

两个训练进程合计宿主机内存约 75–80 GiB（上一轮实测主进程峰值约 58 GiB）。

## 3. 变量

```bash
cd /home/oycy/MMDD
RUN=work/stage1_entitables_r2_fullkd_s13       # 新 run 目录；必须是尚不存在的路径
GPU=0                                          # 主 GPU（物理 index）
SIDE_GPU=1                                     # 第二张 GPU（物理 index，与主 GPU 同型号）
S="conda run --no-capture-output -n MMDD python src/run_stage1.py"
DATASET=output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9
OUT=${RUN}_teacher_sp                          # 续训主臂输出目录（第二部分）
T="conda run --no-capture-output -n MMDD python src/continue_teacher_on_student_pool.py"
```

## 4. 第一部分：正式 run

每一步完成后 `cat $RUN/PHASE_STATUS.json`，`status` 应为下表所列的值。

| # | 命令 | 设备 | 完成标志 |
|---|---|---|---|
| 1 | `$S init --run-root $RUN --dataset-root $DATASET --features-dir work/stage1_features/entitables --gpu $GPU --side-gpu $SIDE_GPU --seeds 13` | CPU | 生成 `$RUN/protocol.json`；打印的 `hardware` 含 `side_uuid` 且 `gpu_processes` 为 2；打印 `existing encoding found; skip build-data and encode` |
| 1b | `python3 -c "import json; s=json.load(open('$RUN/protocol.json'))['student']; print(s['kd_top_k'], s['C1']['P_lr'], s['C1']['R_lr'])"` | CPU | 打印 `0 <C1.P_lr> <C1.R_lr>`，与第 2 节一致 |
| 2 | `$S lock --run-root $RUN` | CPU（约 5 分钟） | `SOURCE_AND_CACHE_LOCKED_PENDING_NUMERIC_PROBE` |
| 3 | `$S verify-features --run-root $RUN` | 主 GPU | `PASS_READY_FOR_REFERENCE_TESTS` |
| 4 | `$S prepare --run-root $RUN` | CPU | `PASS_READY_FOR_VALIDATION` |
| 5 | `$S validate --run-root $RUN` | CPU | `PASS_READY_FOR_SMOKE`；`$RUN/tests/integration/VALIDATION_RECEIPT.json` 的 `status` 为 `PASS` |
| 6 | `$S smoke --run-root $RUN`（后台，见下） | 主 GPU | `PASS_READY_FOR_FORMAL`；`$RUN/tests/smoke/LATEST.json` 的 `status` 为 `PASS` |
| 7 | `$S train` 与 `$S train-side` 两个进程同时启动（见下） | 主 GPU + 第二张 GPU | `COMPLETE_PASS` 或 `COMPLETE_PERFORMANCE_GATES_FAILED`（两者都表示跑完） |

复用已有特征，不执行 `build-data` 和 `encode`。除这两步外不要跳过任何步骤。

```bash
mkdir -p $RUN/logs
setsid nohup $S smoke --run-root $RUN > $RUN/logs/smoke.log 2>&1 < /dev/null &
```

smoke 通过后，紧接着启动两个进程（先后顺序无所谓）：

```bash
setsid nohup $S train      --run-root $RUN > $RUN/logs/train.log      2>&1 < /dev/null & echo $! > $RUN/logs/train.pid
setsid nohup $S train-side --run-root $RUN > $RUN/logs/train_side.log 2>&1 < /dev/null & echo $! > $RUN/logs/train_side.pid
```

**分工。** `train`（主 GPU）跑 Native 主线：Raw 池 → TA → TB_CQET → Native C1 → C2 共享图 → Native C2 SUP/KD → dev 评估。
`train-side`（第二张 GPU）跑其余：TA 结束后训 TB_QT，然后 QT C1 与选点、teacher 轨迹，等 C2 图落盘后训 QT C2 并选点，
全局冻结后做 test 评估。日志里的 `[wait] ...` 是互相等待，正常。

**轮询：**

```bash
date -u; tail -n 5 $RUN/logs/train.log $RUN/logs/train_side.log
cat $RUN/PHASE_STATUS.json
cat $RUN/processes/main.json $RUN/processes/side.json        # status：RUNNING / FAILED / COMPLETE
tail -n 3 $RUN/seed13/timing/stages.jsonl 2>/dev/null        # 每完成一个阶段追加一行
ls $RUN/seed13/selections/ 2>/dev/null
ps -p $(cat $RUN/logs/train.pid) >/dev/null && echo train running || echo train exited
ps -p $(cat $RUN/logs/train_side.pid) >/dev/null && echo side running || echo side exited
```

**里程碑**（出现时各报告一次，带 UTC 时间）：TA 完成（`$RUN/seed13/TA/POST_RUN.attempt_*.json` 有 `status: SUCCESS`）→
TB_CQET 完成 → **Native C1 选点**（`selections/NATIVE_C1.json` 出现，**立刻按 6.2 节汇报 C1 的数字**）→
学生训练结束（`NATIVE_C2_SUP`、`NATIVE_C2_KD` 都有 SUCCESS 的 POST_RUN）→ **学生选点**（`selections/NATIVE_C2_COMMON.json`，
**立刻按 6.3 节汇报**）→ 进入评估（`GLOBAL_SELECTION_FREEZE.json` 出现）→ side 结束（`processes/side.json` 为 `COMPLETE`）→
run 结束（`PHASE_STATUS.json` 的 `status` 以 `COMPLETE_` 开头，`processes/main.json` 为 `COMPLETE`）。

**中断与恢复。** 任一进程挂掉后，另一个会在下一次等待时报 `... died before producing ...` 或 `... is FAILED ...` 并退出。
先把两份日志复制为 `*.failed.<UTC时间>`，报告 `processes/*.json` 的 `error` 和两份日志最后 50 行，然后用**同一组命令**把两个
进程都重新启动一次（已完成阶段和选点会被复用）。第二次仍失败就停下来报告，等用户指示。

**耗时（上一轮实测）：** Raw 池约 1 小时；TA 76 分钟；TB_CQET 58 分钟；C1 1 分钟 + 选点约 50 分钟；C2 建图约 2 小时；
C2 SUP 5 分钟、KD 7 分钟 + 选点约 1 小时；dev 评估约 1 小时。上一轮 06:45 启动、16:10 结束，共 9.4 小时。

## 5. 第二部分：Teacher 续训（run 结束后）

### 5.1 为什么要等两个进程都退出

`train-side` 一直占着 GPU1，直到 test 评估结束、`processes/side.json` 变为 `COMPLETE` 才退出。续训显存峰值约 17–19 GiB，
GPU1 被挤爆会让正式 run 死在收尾阶段。**两个进程都退出后才允许启动。**

### 5.2 启动条件（全部满足才启动）

```bash
python3 - <<'EOF'
import json, os, subprocess
run = "work/stage1_entitables_r2_fullkd_s13"
ok = True
for role in ("main", "side"):
    s = json.load(open(f"{run}/processes/{role}.json"))
    alive = os.path.exists(f"/proc/{s['pid']}")
    print(role, s["status"], "alive" if alive else "exited")
    ok &= s["status"] == "COMPLETE" and not alive
sel = f"{run}/seed13/selections/NATIVE_C2_COMMON.json"
if os.path.exists(sel):
    d = json.load(open(sel))
    pt = next(p for p in d["points"] if p["fraction"] == d["selected_fraction"])
    r10 = pt["KD_metrics"]["Direct_ANN_R10"]
    print("selected KD fraction", d["selected_fraction"], "Direct_ANN_R10", round(r10, 4),
          "checkpoint exists", os.path.exists(d["KD_checkpoint"]))
    ok &= os.path.exists(d["KD_checkpoint"]) and r10 >= 0.35
else:
    print("NATIVE_C2_COMMON.json missing"); ok = False
ok &= os.path.exists(f"{run}/seed13/TB_CQET/checkpoints/end.pt")
gpu = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.used", "--format=csv,noheader"], text=True).strip()
apps = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
print("GPU1:", gpu, "| compute apps:", apps or "none")
ok &= apps == ""
avail = int(subprocess.check_output(["free", "-g"], text=True).splitlines()[1].split()[6])
print("host available GiB", avail); ok &= avail >= 40
print("LAUNCH" if ok else "DO NOT LAUNCH")
EOF
```

KD 学生 `Direct_ANN_R10 < 0.35` 说明学生配方没生效或崩了：**报告数字，等用户指示**，不要启动。

### 5.3 主臂（学生池列表）

脚本和上一轮完全相同，已经在本机跑通过（含 16 查询冒烟），本轮不再冒烟。

```bash
mkdir -p $OUT
setsid nohup $T --run-root $RUN --out-dir $OUT --gpu 1 > $OUT/run.log 2>&1 < /dev/null &
echo $! > $OUT/run.pid
```

启动后 2 分钟再确认进程在 GPU1 上：

```bash
ps -p $(cat $OUT/run.pid) && nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

阶段与上一轮实测耗时：挖掘（`[mine] ...`）25 分钟 → 续训（`[TB_CQET] epoch=1/1 step=k/1579 ...`）61 分钟，显存峰值 16–19 GiB
→ 评估（`[evaluate] ...`）约 13 分钟 → 打印 JSON 表并退出。完成标志：`$OUT/eval/dev/SUMMARY.json` 出现。

轮询：

```bash
date -u; tail -n 3 $OUT/run.log; ps -p $(cat $OUT/run.pid) >/dev/null && echo running || echo exited
tail -n 1 $OUT/train.jsonl 2>/dev/null | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print({k:r[k] for k in ('step','loss','direct','path','support','grad_norm_preclip','teacher_candidate_chunk','teacher_backward_mode','gpu_peak_allocated_bytes')}, 'oom_events', len(r['oom_events']))"
```

`oom_events` 非空、`teacher_backward_mode` 变 `two_pass`、`teacher_candidate_chunk` 变小都是内置显存退让，不是错误。
进程意外退出且无 `SUMMARY.json`：把 `run.log` 复制为 `run.log.failed.<UTC>`，报告最后 50 行，用同一条命令重启一次
（已完成的挖掘 / 训练 / 评估步骤会被跳过）。第二次仍失败就停下来报告。

### 5.4 对照臂（主臂结束后执行；用户没叫停就默认跑）

同一起点、同样 1 个 epoch，列表换回本 run 的 Raw 列表，用来把"多训一个 epoch"与"换成学生池列表"分开：

```bash
mkdir -p ${OUT}_raw2
setsid nohup $T --run-root $RUN --out-dir ${OUT}_raw2 --gpu 1 \
  --records $RUN/seed13/training_records/TB_SHARED.jsonl.gz > ${OUT}_raw2/run.log 2>&1 < /dev/null &
echo $! > ${OUT}_raw2/run.pid
```

没有挖掘阶段，上一轮实测约 1 小时 15 分钟。监视方式同 5.3。

## 6. 汇报内容

### 6.1 正式 run 的结论

`$RUN/PHASE_STATUS.json` 全文；`$RUN/reports/RESULTS.md` 全文（两张表都要）。

### 6.2 C1 是否还被丢弃（`selections/NATIVE_C1.json` 出现时立刻汇报）

取 `selected_fraction`，以及每个 `points[*]` 的 `fraction`、`metrics.{Direct_ANN_R10, C150_target_coverage, D150_target_coverage,
E_target_coverage, U_target_coverage}`、`parameter_drift.R_QT_relative_to_identity`。列成一张表（行是 fraction）。
参照见第 7.1 节。`selected_fraction` 再次为 0，或任一 fraction 的 `E_target_coverage` 低于 0.60，都要明确标出。

### 6.3 学生是否在学、KD 是否超过 SUP（`selections/NATIVE_C2_COMMON.json` 出现时立刻汇报）

取 `selected_fraction`，以及每个 `points[*]` 的 `fraction`、`SUP_metrics.{Direct_ANN_R10, C150_target_coverage, E_target_coverage}`、
`KD_metrics.{Direct_ANN_R10, C150_target_coverage, E_target_coverage}`。列成一张表。参照见第 7.2 节。

漂移监控：`$RUN/seed13/NATIVE_C2_SUP/train.attempt_*.jsonl` 与 `NATIVE_C2_KD/train.attempt_*.jsonl` 首行与末行的
`direct_sup_loss`、`evidence_sup_loss`、`direct_kd_loss`、`evidence_kd_loss`、`sigma1_R_QT_minus_I`、`sigma2_R_QT_minus_I`、
`lr_factor`、各 `*_denominator`。`sigma1/sigma2 > 2` 要标出；`direct_sup_loss` 停在 log(候选数) ≈ 6.9 附近要标出。
KD 臂的 `direct_kd_loss` 本轮应当明显低于上一轮的 1.1（上一轮那个值是 rank-mass 尾项），末行若仍 ≥ 1.0 要标出。

### 6.4 证据叙事指标

`$RUN/seed13/eval/{dev,test}/SUMMARY.json` 的 `contrasts` 块：把每一项的 `mean_delta_pp`、`ci_95`、`wlt` 原样列出，
重点是 `KD_minus_SUP_student_Direct_R10_overall`、`KD_minus_SUP_E_target_coverage_overall`、`KD_minus_SUP_same_TB_CQET_overall`、
`evidence_CQET_Real_minus_f0_{overall,implicit}`、`content_CQET_Real_minus_Swap_implicit`、`CQET_Real_minus_TB_QT_overall`。
另取 `raw / native_sup / native_kd` 三个块下 `teacher.TB_CQET.{f0,Real}.overall.R@10`（Teacher 在三种池上的数字）。
原样贴出，不要自行解读成"机制成立"。

### 6.5 Teacher 续训（主臂与对照臂各一份）

1. `IDENTITY.json`：`student.path`、`student.state_sha256`、`init_checkpoint.sha256`、`records.count`、`records.source`、
   `config.epochs/lr`、`gpu_uuid`、`git_head`。
2. `records_summary.json` 全文（主臂才有）。
3. `TRAIN_TIMING.json` 全文；`train.jsonl` 首行与末行的 `loss`、`direct`、`path`、`support`、`grad_norm_preclip`、
   `gpu_peak_allocated_bytes`，以及 `oom_events` 非空的行数。
4. **验收表**：`eval/dev/SUMMARY.json` → `generators.native_kd` 与 `generators.raw`，各取 `candidate.overall` 的
   `Direct_ANN_R10`、`C150_target_coverage`；`teacher.{init,half,end}.{f0,Real,Swap}` 的 `overall / implicit / explicit`；
   `contrasts` 里 `end.Real_minus_student_Direct_ANN_R10`、`end.Real_minus_init.Real`、`end.f0_minus_init.f0`、
   `half.Real_minus_init.Real` 的 `mean_delta_pp`、`ci_95`、`wlt`。行是 `student / init / half / end`，
   列是 `native_kd f0 | native_kd Real | raw f0 | raw Real`。
5. **一致性核对**：`generators.native_kd.teacher.init` 的 f0/Real/Swap `overall` 必须与正式 run `eval/dev/SUMMARY.json`
   的 `native_kd.teacher.TB_CQET.{f0,Real,Swap}.overall.R@10` 逐位相同；`generators.raw.teacher.init` 对应 `raw.teacher.TB_CQET...`。
   `generators.native_kd.pools` 应指向 `$RUN/seed13/eval/dev/native_kd`。不相同要明确标出。

### 6.6 运行信息

两个训练进程和两条续训各自的开始/结束 UTC 时间、GPU UUID；`$RUN/seed13/timing/stages.jsonl` 全文；`git status --short`
与 `git log --oneline -1`；是否发生过重启及原因。

## 7. 参照数字与读法（参照不是保证；只陈述，不下结论）

### 7.1 C1

C1 单独学习率的取值依据是在上一轮 run 的边列表上做的学习率探针（`work/stage1_entitables_abcd_s13_probe/c1_lr/RESULTS.jsonl`，
dev，1198 查询，Native C1，余弦衰减，尺度 20；与正式 run 的 C1 选点用同一套检索与覆盖度量）：

| C1 学习率 P / R | fraction | Direct_ANN_R10 | C150 覆盖 | D150 覆盖 | E 覆盖 | U 覆盖 | R_QT 漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity（fraction 0，上一轮实际选中点） | 0.0 | 0.2551 | 0.5807 | 0.5529 | 0.6834 | 0.7193 | 0.000 |
| 1e-5 / 1e-4 | 0.5 | 0.3088 | 0.6813 | 0.6370 | 0.7305 | 0.8161 | 0.088 |
| 1e-5 / 1e-4 | 1.0 | 0.2667 | 0.6772 | 0.5868 | 0.7308 | 0.8066 | 0.103 |
| 1e-6 / 1e-5（v4.1 旧值） | 1.0 | 0.2668 | 0.5961 | 0.5700 | 0.6976 | 0.7417 | 0.010 |
| 3e-5 / 3e-4 | 1.0 | 0.2628 | 0.5051 | 0.5687 | 0.3555 | 0.6087 | 0.251 |
| 上一轮正式 run 1e-4 / 1e-3（共享值） | 0.25 | 0.2309 | 0.4872 | 0.5626 | 0.3404 | 0.5927 | 0.437 |
| 上一轮正式 run 1e-4 / 1e-3（共享值） | 1.0 | 0.2607 | 0.5478 | 0.6722 | 0.3334 | 0.6646 | 0.587 |

本轮取 **`C1.P_lr = 1e-05`、`C1.R_lr = 0.0001`**（上表中 C150 覆盖最高且 E 覆盖不低于 identity 的一档）。

读法：`selected_fraction > 0` 且选中点的 `E_target_coverage ≥ 0.68`、`C150_target_coverage > 0.581`，说明 C1 这次没有被丢弃。
再退回 fraction 0 也不是程序错误，C2 会像上一轮一样从 identity 起训，KD 对照仍然有效；照实报告。

### 7.2 C2

direct-only 探针（同一份 C2 列表、同一个 Teacher、固定负样本 + 余弦、3 epoch，精确 R@10，与正式 run 的 ANN 数字可差 0–1pp）：

| 配置 | dev 1/3 | dev 2/3 | dev 结束 | test 结束 |
|---|---:|---:|---:|---:|
| SUP | 0.416 | 0.436 | 0.439 | 0.425 |
| SUP + KD，全列表 KL（本轮配置） | 0.453 | 0.470 | 0.465 | 0.451 |
| SUP + KD，top-50（上一轮配置） | 0.407 | 0.419 | 0.428 | 0.412 |

正式 run 还同时训证据分支，所以数字会有出入。参照：KD 学生 Direct_ANN_R10 比 SUP 高 2–3pp 算与探针一致；
KD ≤ SUP 要明确标出。3 epoch 下相邻快照的 dev R@10 波动约 ±2pp，单种子，1pp 以内的差别不要解读。

E 池覆盖：上一轮 KD 比 SUP 低 10.6pp。本轮如果仍低 ≥ 5pp，就只能归到 `evidence_random_negatives` / evidence KD（方案 D），照实报。

### 7.3 Teacher 续训

验收线（诊断文档 7.2）：`native_kd` 池上 `end.Real ≥ 学生 Direct_ANN_R10`，且 `end.Real_minus_init.Real` 的 `ci_95` 不跨 0。
上一轮主臂：+9.3pp [7.0, 11.6]，高出学生 6.0pp [3.4, 8.6]；对照臂：−0.3pp [−1.5, 0.9]。`raw` 池上两臂都掉约 0.8pp
（CI 跨 0），是多训一个 epoch 的副作用，与列表无关。`implicit` 段单独列出 `Real − f0`（上一轮 end：+2.5pp）。

## 8. 这个任务不做的事

- 不改协议、不改源码、不调超参、不多跑 epoch、不换 Teacher 起点。
- 不用续训后的 Teacher 重新蒸馏学生、不改 `teacher_logits_cache`、不跑 Stage 2、不碰上一轮的目录。这些由用户看完报告后决定。
