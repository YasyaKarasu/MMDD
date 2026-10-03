# Stage-1 Teacher 续训（学生池难负例）监视与启动说明（EntiTables，seed 13，GPU1）

给执行者（Gemini）：这是一个"监视 + 择机启动 + 汇报"的任务。先读完全文再动手。
背景见 `docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md` 第 7 节；正在跑的实验见
`docs/stage1_plan_abcd_run_guide.zh-CN.md`。

## 0. 任务目标

1. **监视** GPU0 上正在进行的 Stage-1 run（`work/stage1_entitables_abcd_s13`）的进展，按阶段汇报。
2. **当该 run 的两个进程都结束后**，在 GPU1 上启动 Teacher 续训：用 dev 选出的 Native C2 KD 学生在
   train 查询上做真实的两跳检索，把检索到的候选（含学生挑出的困难负例）组成 T_B 训练列表，从
   `TB_CQET/checkpoints/end.pt` 出发再训 1 个 T_B epoch，然后在**同一个学生的 dev 候选池**上比较
   续训前后的 Teacher 重排 R@10。脚本已写好：`src/continue_teacher_on_student_pool.py`。
3. **汇报**结果（第 7 节）。不要自行解读为"机制成立"，也不要为了让数字好看而改配置重跑。

## 1. 为什么要等整个 run 结束，而不是"学生训完就启动"

这个 run 是双进程的：`train`（GPU0）和 `train-side`（GPU1）。**`train-side` 一直占着 GPU1**：TA 之后它在
GPU1 上训 TB_QT、QT C1/C2、跑 Teacher 轨迹，学生选点冻结之后它还要在 GPU1 上跑整个 **test 评估**
（约 1 小时），直到 `processes/side.json` 变成 `COMPLETE` 才退出。

Teacher 续训在 EntiTables 上的显存峰值约 19–20 GiB（4090 共 24 GiB）。`train-side` 的检索/评估代码
没有 OOM 重试，一旦在 GPU1 上被挤爆，`train-side` 失败，`train` 随之失败，8–10 小时的正式 run 会死在
收尾阶段。所以**只有在两个进程都退出后才允许启动续训**。学生训完到 run 结束大约再隔 1.5–2 小时。

## 2. 规则（必须遵守）

- 所有命令在 `/home/oycy/MMDD` 下执行，conda 环境 `MMDD`。时间一律用 `date -u`（UTC）。
- **不要修改 `src/` 和 `tests/` 下的任何文件**。正式 run 把源码哈希绑进了阶段收据，改了会让它拒绝继续。
  遇到代码报错，停下来报告报错与日志路径，不要自行打补丁。
- 不要读取、查看或修改 `.env.openai`。
- 不要删除或覆盖 `work/stage1_features/` 和 `work/stage1_entitables_abcd_s13/` 下的任何内容。续训脚本
  只往 `--out-dir` 写（它会把 run 目录下的 `Z_UNIT_IDENTITY.json` 用完全相同的内容重写一次，这是已知且无害的）。
- **不要重启正式 run**。它失败了就报告，谁来重启由用户决定。
- 不要碰 test 切分：续训脚本只用 train 查询挖候选、只在 dev 上评估。不要给它加 test 相关的参数。
- `train-side` 处于 `RUNNING` 时，**GPU1 上不要启动任何东西**。
- 同一时间只跑一个续训进程。
- 本机首次 CUDA 初始化偶尔报 `CUDA driver initialization failed`（`cuInit` 返回 3）。不是权限问题，原样重跑一次即可。

## 3. 变量

```bash
cd /home/oycy/MMDD
RUN=work/stage1_entitables_abcd_s13
OUT=work/stage1_entitables_abcd_s13_teacher_sp          # 续训输出目录，必须在 run 目录之外
GPU1_UUID=GPU-b1fcd6e2-b542-a1b0-8656-202feee8afc7      # 物理 index 1；与 $RUN/protocol.json 的 hardware.side_uuid 一致
S="conda run --no-capture-output -n MMDD python src/continue_teacher_on_student_pool.py"
```

## 4. 监视阶段（run 结束前）

每 10–15 分钟轮询一次，**不要原地阻塞**：

```bash
date -u
cat $RUN/PHASE_STATUS.json
cat $RUN/processes/main.json $RUN/processes/side.json          # status：RUNNING / FAILED / COMPLETE
tail -n 3 $RUN/logs/train.log $RUN/logs/train_side.log
tail -n 3 $RUN/seed13/timing/stages.jsonl 2>/dev/null           # 每完成一个训练阶段追加一行
ls $RUN/seed13/selections/ 2>/dev/null
ps -p $(cat $RUN/logs/train.pid) >/dev/null && echo train running || echo train exited
ps -p $(cat $RUN/logs/train_side.pid) >/dev/null && echo side running || echo side exited
```

日志里的 `[wait] ...` 是两个进程互相等待，属正常现象，不是卡死。

**里程碑**（出现时各报告一次，带 UTC 时间）：

| 里程碑 | 判据 |
|---|---|
| TA 完成 | `$RUN/seed13/TA/POST_RUN.attempt_*.json` 有 `status: SUCCESS` |
| TB_CQET 完成 | 同上，目录 `TB_CQET` |
| Native C1 选点 | `$RUN/seed13/selections/NATIVE_C1.json` 出现 |
| **学生训练结束** | `NATIVE_C2_SUP` 与 `NATIVE_C2_KD` 两个目录都有 SUCCESS 的 `POST_RUN` |
| 学生选点完成 | `$RUN/seed13/selections/NATIVE_C2_COMMON.json` 出现（C2 结束后约 30 分钟） |
| 进入评估 | `$RUN/GLOBAL_SELECTION_FREEZE.json` 出现，`PHASE_STATUS.json` 的 `status` 为 `FROZEN_EVALUATION` |
| side 结束 | `processes/side.json` 为 `COMPLETE`（test 评估结束，GPU1 释放） |
| run 结束 | `PHASE_STATUS.json` 的 `status` 以 `COMPLETE_` 开头；`processes/main.json` 为 `COMPLETE` |

run 于 2026-10-02 06:45 UTC 启动，指南估计总耗时 8–10 小时（粗估，未实测）。

**学生选点完成后立刻汇报**（这是续训的输入，也是用户最关心的早期信号）。从
`$RUN/seed13/selections/NATIVE_C2_COMMON.json` 取 `selected_fraction`，以及每个 `points[*]` 的
`fraction`、`SUP_metrics.Direct_ANN_R10`、`KD_metrics.Direct_ANN_R10`、`SUP_metrics.C150_target_coverage`、
`KD_metrics.C150_target_coverage`。参照：旧配方 dev Direct R@10 ≈ 0.27；诊断文档的探针 SUP ≈ 0.43、KD ≈ 0.47。

**任一进程 `FAILED`**：立刻把 `processes/*.json` 里的 `error`、两份日志的最后 50 行、`$RUN/ERROR_LEDGER.jsonl`
的最后几行报告给用户，然后停止本任务（不启动续训，不重启 run）。

## 5. 启动条件（全部满足才启动）

```bash
python3 - <<'EOF'
import json, os, subprocess
run = "work/stage1_entitables_abcd_s13"
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
ok &= apps == "" and gpu.startswith("GPU-b1fcd6e2")
avail = int(subprocess.check_output(["free", "-g"], text=True).splitlines()[1].split()[6])
print("host available GiB", avail); ok &= avail >= 40
print("LAUNCH" if ok else "DO NOT LAUNCH")
EOF
```

逐条含义：

1. `main` 和 `side` 都是 `COMPLETE` 且进程已退出。`FAILED` 走第 4 节的失败处理。
2. `NATIVE_C2_COMMON.json` 存在，选中的 KD 学生 checkpoint 在盘上，且其 dev `Direct_ANN_R10 ≥ 0.35`。
   低于 0.35 说明学生配方没生效或崩了，在它的池上续训 Teacher 没有意义：**报告数字，等用户指示**，不要启动。
3. `TB_CQET/checkpoints/end.pt` 存在（续训起点）。
4. GPU1 上没有计算进程。
5. 宿主机 `available ≥ 40 GiB`（脚本要把 4.3 GiB 的冻结特征和 1.2 万条 train 查询的检索池放进内存）。

## 6. 启动与监视续训

### 6.1 先做 16 个查询的冒烟（约 10 分钟）

目的：在 EntiTables 规模的索引和显存上验证整条路径能跑通，再投入 2–3 小时的正式续训。

```bash
mkdir -p ${OUT}_smoke16
setsid nohup $S --run-root $RUN --out-dir ${OUT}_smoke16 --gpu 1 --limit 16 > ${OUT}_smoke16/run.log 2>&1 < /dev/null &
echo $! > ${OUT}_smoke16/run.pid
```

完成标志：`${OUT}_smoke16/eval/dev/SUMMARY.json` 出现，`run.log` 末尾打印了一张以 `"native_kd"` 和 `"raw"`
为键的表。失败则报告 `run.log` 最后 50 行并停止。冒烟目录用完不用删。

### 6.2 正式续训（主臂：学生池列表）

```bash
mkdir -p $OUT
setsid nohup $S --run-root $RUN --out-dir $OUT --gpu 1 > $OUT/run.log 2>&1 < /dev/null &
echo $! > $OUT/run.pid
```

**必须用 `setsid nohup ... &` 脱离会话**，否则你的回合结束时进程会被杀掉。启动后 2 分钟内确认：

```bash
ps -p $(cat $OUT/run.pid) && nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

进程必须出现在 `GPU-b1fcd6e2...` 上。脚本内部会把 `CUDA_VISIBLE_DEVICES` 固定为该 UUID 并核对，不匹配会直接报
`BLOCKED_GPU_IDENTITY` 退出。

**阶段与预期耗时**（粗估；`run.log` 里的标记）：

| 阶段 | 日志标记 | 产物 | 预期 |
|---|---|---|---|
| 加载运行时 | 无输出，约 2–3 分钟 | `IDENTITY.json`（在挖掘之后写） | |
| 挖掘（学生检索 train 查询） | `[mine] N train queries with the KD Student` | `records_summary.json`、`records.jsonl.gz`、`indices/` | 建 3 个 HNSW 索引约 3–5 分钟，之后约 0.15 s/查询，共 30–45 分钟 |
| 续训 | `[TB_CQET] epoch=1/1 step=k/K loss=...`（K ≈ 1580） | `checkpoints/{init,half,end}.pt`、`train.jsonl`、`TRAIN_TIMING.json` | 约 1–1.5 小时；显存峰值约 19–20 GiB |
| 评估 | `[evaluate] native_kd dev pools from ...`、`[evaluate] raw dev pools from ...` | `eval/dev/{native_kd,raw}/`、`eval/dev/SUMMARY.json` | 约 20–40 分钟 |
| 结束 | 打印 JSON 表，进程退出 | | |

每 10–15 分钟轮询：

```bash
date -u; tail -n 3 $OUT/run.log; ps -p $(cat $OUT/run.pid) >/dev/null && echo running || echo exited
tail -n 1 $OUT/train.jsonl 2>/dev/null | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print({k:r[k] for k in ('step','loss','direct','path','support','grad_norm_preclip','teacher_candidate_chunk','teacher_backward_mode','gpu_peak_allocated_bytes')}, 'oom_events', len(r['oom_events']))"
```

- `oom_events` 非空、`teacher_backward_mode` 变成 `two_pass` 或 `teacher_candidate_chunk` 变小，都是脚本内置的
  显存退让，**不是错误**，只会变慢。
- 进程意外退出（`exited` 且 `SUMMARY.json` 不存在）：先把 `run.log` 复制为 `run.log.failed.<UTC时间>`，报告最后 50 行，
  然后用**同一条命令**重启一次。脚本会跳过已完成的步骤：`records.jsonl.gz` 已有就不再挖掘；`checkpoints/end.pt`
  已有就不再训练（半途的 checkpoints 会被改名为 `checkpoints.failed.<ts>` 后从头训）；`SUMMARY.json` 已有就直接结束。
  第二次仍失败就停下来报告。

### 6.3 对照臂（主臂结束后执行；用户没叫停就默认跑）

同一个起点、同样的 1 个 epoch，但列表换回正式 run 原来的 Raw 列表（`TB_SHARED.jsonl.gz`）。它把
"多训了一个 epoch"和"换成学生池列表"两个因素分开：主臂比对照臂好多少，才是学生池难负例的贡献。

```bash
mkdir -p ${OUT}_raw2
setsid nohup $S --run-root $RUN --out-dir ${OUT}_raw2 --gpu 1 \
  --records $RUN/seed13/training_records/TB_SHARED.jsonl.gz > ${OUT}_raw2/run.log 2>&1 < /dev/null &
echo $! > ${OUT}_raw2/run.pid
```

没有挖掘阶段，约 1.5–2 小时。监视方式同 6.2。

## 7. 汇报内容

### 7.1 正式 run 的状态

`$RUN/PHASE_STATUS.json` 全文；`$RUN/reports/RESULTS.md` 的路径；第 4 节里学生选点那组数字；
`$RUN/seed13/eval/dev/SUMMARY.json` 的 `narrative.native_kd.overall` 块原样贴出（里面有学生自己的
`Direct_ANN_R10`、`C150_target_coverage` 和原 Teacher 的 `TB_CQET.{f0,Real,Swap}.R@10`）。

### 7.2 续训（主臂与对照臂各一份）

1. `IDENTITY.json`：`student.path`、`student.state_sha256`、`init_checkpoint.sha256`、`records.count`、
   `records.source`、`config.epochs/lr`、`gpu_uuid`、`git_head`。
2. `records_summary.json` 全文（主臂才有）：`mean_targets`、`mean_paths`、`U_target_coverage`、
   `D150_target_coverage`、`C150_target_coverage`、`Direct_ANN_R10` 都是 train 查询上学生自己的池。
   和正式 run 的 Raw 列表对比：`TB_SHARED.jsonl.gz` 每条平均约 620 个 target、850 条路径（v4.1 实测）。
3. `TRAIN_TIMING.json` 全文；`train.jsonl` 首行与末行的 `loss`、`direct`、`path`、`support`、各 `*_denominator`、
   `grad_norm_preclip`、`gpu_peak_allocated_bytes`，以及整个文件里 `oom_events` 非空的行数。
4. **验收表**：`eval/dev/SUMMARY.json` → `generators.native_kd` 和 `generators.raw`，各取
   - `candidate.overall` 的 `queries`、`Direct_ANN_R10`、`C150_target_coverage`；
   - `teacher.{init,half,end}.{f0,Real,Swap}` 的 `overall / implicit / explicit`；
   - `contrasts` 里 `end.Real_minus_student_Direct_ANN_R10`、`end.Real_minus_init.Real`、`end.f0_minus_init.f0`、
     `half.Real_minus_init.Real` 的 `mean_delta_pp`、`ci_95`、`wlt`。
   用一张表列出来，行是 `student / init / half / end`，列是 `native_kd f0 | native_kd Real | raw f0 | raw Real`。
5. **一致性核对**：`generators.native_kd.teacher.init` 的 f0/Real/Swap `overall` 必须与 7.1 里正式 run `SUMMARY.json` 的
   `narrative.native_kd.overall.TB_CQET.{f0,Real,Swap}.R@10` 逐位相同；`generators.raw.teacher.init` 则对应正式 run
   `SUMMARY.json` 的 `raw.teacher.TB_CQET.{f0,Real,Swap}.overall.R@10`。`generators.native_kd.pools` 应指向
   `$RUN/seed13/eval/dev/native_kd`。不相同要明确标出。

### 7.3 怎么读（只陈述，不下结论）

- 这次实验的验收线（诊断文档 7.2）：在 `native_kd` 池上，续训后 Teacher 的 `end.Real` 要
  **≥ 学生自己的 `Direct_ANN_R10`**，并且 `end.Real_minus_init.Real` 的 `ci_95` 不跨 0。两条都满足才算"Teacher
  在真实学生池上重新领先"。
- `raw` 池上 `end` 相对 `init` 的变化是副作用监控：下降说明 Teacher 对 Raw 分布有遗忘，照实报。
- 主臂与对照臂的 `end.Real` 之差（在 `native_kd` 池上）是列表分布的净贡献。
- `implicit` 段是论文叙事里证据机制要起作用的那部分，单独列出 `Real − f0`（`end` 与 `init` 各一份，可从
  `teacher` 块里相减）。
- 不要把这些读成"机制成立/不成立"，原样给数字和区间。

### 7.4 运行信息

两条续训各自的开始/结束 UTC 时间、输出目录、`git status --short` 的输出（应当只有本说明和续训脚本相关的改动）、
是否发生过重启及原因。

## 8. 这个任务不做的事

- 不用新 Teacher 重新蒸馏学生、不改 `teacher_logits_cache`、不跑 test、不碰 Stage 2。这些由用户看过 7.2 的表后决定。
- 不调超参数、不多跑 epoch、不换起点（起点固定为 `TB_CQET/checkpoints/end.pt`；从 `TA/checkpoints/epoch2.pt`
  重训 T_B 是另一个待定方案，没有用户指示不要做）。
