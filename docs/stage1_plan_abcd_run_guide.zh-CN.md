# Stage-1 方案 A+B+C+D 实验运行说明（EntiTables，seed 13，双卡）

给执行者：按顺序执行下面的命令，每一步检查完成标志再进入下一步。背景见
`docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md`。实现说明见 `src/README.md`
的 “Stage-1 main flow” 一节（其中 “Two GPUs” 一段说明了双卡分工）。

## 0. 规则（必须遵守）

- 所有命令都在仓库根目录 `/home/oycy/MMDD` 下执行，使用 conda 环境 `MMDD`。
- **运行期间不要修改 `src/` 和 `tests/` 下的任何文件。** `validate`、`smoke`、`train`、`train-side`
  会绑定源码哈希，改了源码，后续步骤会拒绝执行。遇到代码错误时，停下来报告报错与日志路径，不要自行打补丁。
- 不要读取、查看或修改 `.env.openai`。本实验不需要它。
- 不要删除或覆盖 `work/stage1_features/` 下的任何内容。它是多个 run 共享的冻结特征，本实验只读。
- 只跑本文这一个 run，不要同时跑其它 run。
- 训练里 gate 判定为 FAIL 是研究结论，不是程序错误。照常完成并如实报告，不要为了让 gate 通过而改配置重跑。

## 1. 运行前检查

```bash
cd /home/oycy/MMDD
nvidia-smi --query-gpu=index,name,memory.used --format=csv   # 两张 4090 都应空闲（memory.used 接近 0）
free -g                                                       # available 必须 ≥ 90 GiB；机器上不要有其它大内存任务
git status --short                                            # 记录当前源码状态，写进最终报告
ls work/stage1_features/entitables/features/z/z.f32.npy       # 冻结特征必须存在
```

两个训练进程合计的宿主机内存估计在 75–80 GiB 左右（主进程峰值约 58 GiB，side 进程另加 15–20 GiB，后者为估计值）。

本机首次 CUDA 初始化偶尔会报 `CUDA driver initialization failed`（`cuInit` 返回 3）。这不是权限问题，
原样重跑一次该命令即可。

## 2. 变量

```bash
cd /home/oycy/MMDD
RUN=work/stage1_entitables_abcd_s13          # 新的 run 目录；必须是尚不存在的路径
GPU=0                                        # 主 GPU（物理 index）
SIDE_GPU=1                                   # 第二张 GPU（物理 index，必须与主 GPU 同型号）
S="conda run --no-capture-output -n MMDD python src/run_stage1.py"
DATASET=output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9
```

## 3. 步骤

每一步完成后都查看 `cat $RUN/PHASE_STATUS.json`，其中 `status` 应为下表所列的值。

| # | 命令 | 设备 | 完成标志（`PHASE_STATUS.json` 的 `status` 或文件） |
|---|---|---|---|
| 1 | `$S init --run-root $RUN --dataset-root $DATASET --features-dir work/stage1_features/entitables --gpu $GPU --side-gpu $SIDE_GPU --seeds 13` | CPU | 生成 `$RUN/protocol.json`，打印的 `hardware` 中含 `side_uuid` 且 `gpu_processes` 为 2，并打印 `existing encoding found; skip build-data and encode` |
| 2 | `$S lock --run-root $RUN` | CPU（约 5 分钟，32 进程） | `SOURCE_AND_CACHE_LOCKED_PENDING_NUMERIC_PROBE` |
| 3 | `$S verify-features --run-root $RUN` | 主 GPU | `PASS_READY_FOR_REFERENCE_TESTS` |
| 4 | `$S prepare --run-root $RUN` | CPU | `PASS_READY_FOR_VALIDATION` |
| 5 | `$S validate --run-root $RUN` | CPU | `PASS_READY_FOR_SMOKE`；`$RUN/tests/integration/VALIDATION_RECEIPT.json` 的 `status` 为 `PASS` |
| 6 | `$S smoke --run-root $RUN`（按下面的后台方式启动） | 主 GPU | `PASS_READY_FOR_FORMAL`；`$RUN/tests/smoke/LATEST.json` 的 `status` 为 `PASS` |
| 7 | `$S train` 和 `$S train-side` **两个进程同时**启动（见下） | 主 GPU + 第二张 GPU | `COMPLETE_PASS` 或 `COMPLETE_PERFORMANCE_GATES_FAILED`（两者都表示跑完） |

本实验复用已有特征，所以不执行 `build-data` 和 `encode`。除这两步外，不要跳过任何步骤。

**`smoke`、`train`、`train-side` 必须脱离当前会话在后台运行。** 否则执行者回合结束时进程会被一并杀掉：

```bash
mkdir -p $RUN/logs
setsid nohup $S smoke --run-root $RUN > $RUN/logs/smoke.log 2>&1 < /dev/null &
```

第 7 步：smoke 通过后，紧接着启动两个进程，先后顺序无所谓：

```bash
setsid nohup $S train      --run-root $RUN > $RUN/logs/train.log      2>&1 < /dev/null & echo $! > $RUN/logs/train.pid
setsid nohup $S train-side --run-root $RUN > $RUN/logs/train_side.log 2>&1 < /dev/null & echo $! > $RUN/logs/train_side.pid
```

**分工。**
- `train`（主 GPU）跑 Native 主线：Raw 池 → TA → TB_CQET → Native C1 → C2 共享图 → Native C2 SUP/KD → dev 评估。
- `train-side`（第二张 GPU）跑其余部分：TA 结束后训练 TB_QT，然后是 QT C1 与选点、teacher 轨迹，等 C2 图落盘后训练 QT C2 并选点，最后在全局冻结后做 test 评估。
- 两个进程在开头和中途会互相等待（日志里出现 `[wait] ...`）。这是正常现象，不是卡死。例如 TA 跑完之前，`train-side` 会一直显示在等 `seed13/TA`。

**定期轮询，不要原地阻塞：**

```bash
tail -n 5 $RUN/logs/train.log $RUN/logs/train_side.log
cat $RUN/PHASE_STATUS.json
cat $RUN/processes/main.json $RUN/processes/side.json         # 每个进程的 status：RUNNING / FAILED / COMPLETE
tail -n 3 $RUN/seed13/timing/stages.jsonl 2>/dev/null         # 每完成一个训练阶段追加一行，含 gpu_uuid
ps -p $(cat $RUN/logs/train.pid) >/dev/null && echo train running || echo train exited
ps -p $(cat $RUN/logs/train_side.pid) >/dev/null && echo side running || echo side exited
```

**中断与恢复。**
- 任一进程挂掉（OOM、报错、机器重启）后，另一个进程会在下一次等待时报
  `the main/side process ... died before producing ...` 或 `... is FAILED ...` 并退出。
- 先保存两份日志，然后用**同一组命令**把两个进程都重新启动。已经成功的阶段和选点会被复用，前提是源码没有改动。
- 只有源码确实改动过，才需要按 `src/README.md` 的说明执行 `amend-source`。这种情况请先报告，不要自行操作。
- `train-side` 正常跑完后 `processes/side.json` 为 `COMPLETE`，进程退出。之后 `train` 还会继续收尾，这也是正常的。

**耗时（粗估，外推，未实测）。** 整个第 7 步预计 8–10 小时，由 `train` 的主线决定：
- Raw 池 1–1.5 小时；
- TA 约 1.2 小时；
- TB_CQET 约 1 小时；
- C1 选点与 C2 建图约 2 小时；
- C2 训练与选点约 1.5–2 小时；
- dev 评估约 1 小时。

`train-side` 的工作量更小，期间会有较长时间在等待。

## 4. 跑完后要汇报的内容

1. **结论**：最终的 `PHASE_STATUS.json`，以及 `$RUN/reports/RESULTS.md` 全文（两张表都要）。
2. **学生是否真的在学**：取 `$RUN/seed13/selections/NATIVE_C2_COMMON.json` 中每个 `points[*]` 的以下字段：
   `fraction`、`SUP_metrics.Direct_ANN_R10`、`KD_metrics.Direct_ANN_R10`、`SUP_metrics.C150_target_coverage`、
   `KD_metrics.C150_target_coverage`，以及 `selected_fraction`。
   - 参照：旧配方的 dev Direct R@10 ≈ 0.27；诊断文档的探针在 direct-only 设置下，SUP ≈ 0.43，KD ≈ 0.47。
   - 这些参照不是保证。≈ 0.27 说明配方没生效；< 0.1 说明崩溃。出现这两种情况都要明确指出。
3. **漂移监控**：查看 C2 训练日志 `$RUN/seed13/NATIVE_C2_SUP/train.attempt_*.jsonl` 和
   `$RUN/seed13/NATIVE_C2_KD/train.attempt_*.jsonl`，报告首行与末行的以下字段：
   `direct_sup_loss`、`evidence_sup_loss`、`direct_kd_loss`、`evidence_kd_loss`、
   `sigma1_R_QT_minus_I`、`sigma2_R_QT_minus_I`、`lr_factor`，以及各 `*_denominator`。
   - `sigma1/sigma2 > 2` 是秩一漂移的信号，要特别标出。
   - `direct_sup_loss` 一直停在 log(候选数) ≈ 6.9 附近，说明学生没在学，也要标出。
4. **证据叙事指标**：取 `$RUN/seed13/eval/{dev,test}/SUMMARY.json` 中的 `narrative` 块和 `contrasts` 块，
   重点是 `evidence_CQET_Real_minus_f0_*`、`CQET_Real_minus_TB_QT_overall`、`KD_minus_SUP_*`、
   `content_CQET_Real_minus_Swap_implicit`。原样贴出数值即可，不要自行解读成“机制成立”。
5. **运行信息**：
   - 两个进程各自的开始与结束时间、两张 GPU 的 UUID；
   - `$RUN/seed13/timing/stages.jsonl` 全文（每个阶段的 `wall_seconds` 和 `gpu_uuid`）；
   - `git status --short` 的输出；
   - 如有失败重启，列出原因和日志路径。
