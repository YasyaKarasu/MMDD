# Stage-1 方案 A+B+C+D 实验运行说明（EntiTables，seed 13）

给执行者：按顺序执行下面的命令，每一步检查完成标志再进入下一步。背景见
`docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md`。实现与开关说明见 `src/README.md`
的 “Stage-1 main flow” 一节。

## 0. 规则（必须遵守）

- 所有命令都在仓库根目录 `/home/oycy/MMDD` 下执行，使用 conda 环境 `MMDD`。
- **运行期间不要修改 `src/` 和 `tests/` 下的任何文件。** `validate`、`smoke`、`train` 会绑定源码哈希，
  改了源码，后续步骤会拒绝执行。遇到代码错误时，停下来报告报错与日志路径，不要自行打补丁。
- 不要读取、查看或修改 `.env.openai`。本实验不需要它。
- 不要删除或覆盖 `work/stage1_features/` 下的任何内容。它是多个 run 共享的冻结特征，本实验只读。
- 只跑本文列出的 run。消融实验（第 5 节）只在用户明确要求时执行。
- 训练里 gate 判定为 FAIL 是研究结论，不是程序错误。照常完成并如实报告，不要为了让 gate 通过而改配置重跑。

## 1. 运行前检查

```bash
cd /home/oycy/MMDD
nvidia-smi --query-gpu=index,name,memory.used --format=csv   # 选一张空闲的 4090，记下 index
free -g                                                       # available 建议 ≥ 80 GiB；机器上不要有其它大内存任务
git status --short                                            # 记录当前源码状态，写进最终报告
ls work/stage1_features/entitables/features/z/z.f32.npy       # 冻结特征必须存在
```

本机首次 CUDA 初始化偶尔会报 `CUDA driver initialization failed`（`cuInit` 返回 3）。这不是权限问题，
原样重跑一次该命令即可。

## 2. 变量

```bash
cd /home/oycy/MMDD
RUN=work/stage1_entitables_abcd_s13          # 新的 run 目录；必须是尚不存在的路径
GPU=0                                        # 第 1 步选定的物理 GPU index
S="conda run --no-capture-output -n MMDD python src/run_stage1.py"
DATASET=output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9
```

## 3. 步骤

每一步完成后都查看 `cat $RUN/PHASE_STATUS.json`，其中 `status` 应为下表所列的值。

| # | 命令 | 设备 | 完成标志（`PHASE_STATUS.json` 的 `status` 或文件） |
|---|---|---|---|
| 1 | `$S init --run-root $RUN --dataset-root $DATASET --features-dir work/stage1_features/entitables --gpu $GPU --seeds 13` | CPU | 生成 `$RUN/protocol.json`，并打印 `existing encoding found; skip build-data and encode` |
| 2 | `$S lock --run-root $RUN` | CPU | `SOURCE_AND_CACHE_LOCKED_PENDING_NUMERIC_PROBE` |
| 3 | `$S verify-features --run-root $RUN` | GPU | `PASS_READY_FOR_REFERENCE_TESTS` |
| 4 | `$S prepare --run-root $RUN` | CPU | `PASS_READY_FOR_VALIDATION` |
| 5 | `$S validate --run-root $RUN` | CPU | `PASS_READY_FOR_SMOKE`；`$RUN/tests/integration/VALIDATION_RECEIPT.json` 的 `status` 为 `PASS` |
| 6 | `$S smoke --run-root $RUN`（按下面的后台方式启动） | GPU | `PASS_READY_FOR_FORMAL`；`$RUN/tests/smoke/LATEST.json` 的 `status` 为 `PASS` |
| 7 | `$S train --run-root $RUN`（按下面的后台方式启动） | GPU | `COMPLETE_PASS` 或 `COMPLETE_PERFORMANCE_GATES_FAILED`（两者都表示跑完） |
| 8（可选，用户需要 Stage-2 输入时） | `$S export --run-root $RUN` | GPU | `$RUN/stage2_handoff/stage1_gate.json` |

**不要跳过 `build-data` / `encode` 以外的任何步骤。** 本实验复用已有特征，所以不执行 `build-data` 和 `encode`。

**`smoke` 与 `train` 必须脱离当前会话在后台运行。** 否则执行者回合结束时进程会被一并杀掉：

```bash
mkdir -p $RUN/logs
setsid nohup $S train --run-root $RUN > $RUN/logs/train.log 2>&1 < /dev/null &
echo $! > $RUN/logs/train.pid
```

（`smoke` 同理，把 `train` 换成 `smoke`，日志写到 `$RUN/logs/smoke.log`。）之后定期轮询，不要原地阻塞：

```bash
tail -n 5 $RUN/logs/train.log
cat $RUN/PHASE_STATUS.json
tail -n 3 $RUN/seed13/timing/stages.jsonl 2>/dev/null     # 每完成一个训练阶段追加一行
ps -p $(cat $RUN/logs/train.pid) >/dev/null && echo running || echo exited
```

**中断与恢复。** `train` 中途失败（例如 OOM、机器重启）时，先保存日志，然后用**同一条命令**重新启动。
已经成功的阶段会被复用，前提是源码没有改动。只有源码确实改动过，才需要按 `src/README.md` 的说明执行
`amend-source`；这种情况请先报告，不要自行操作。

**耗时（粗估，未实测）。**
- TA 两个 epoch 约 1 小时，TB_CQET、TB_LSE、TB_QT 合计约 3 小时。
- 学生 C1/C2 每个阶段是分钟级。
- 每个选点都要做一次 dev 检索和 teacher 重排；每次检索先单线程建 HNSW 索引，约 3 分钟。
- 冻结的 dev/test 评估还要几个小时。
- 整个 `train` 预计十几小时。

## 4. 跑完后要汇报的内容

1. **结论**：最终的 `PHASE_STATUS.json`，以及 `$RUN/reports/RESULTS.md` 全文（两张表都要）。
2. **学生是否真的在学**：取 `$RUN/selections/NATIVE_C2_COMMON.json` 中每个 `points[*]` 的
   `fraction`、`SUP_metrics.Direct_ANN_R10`、`KD_metrics.Direct_ANN_R10`、`SUP_metrics.C150_target_coverage`、
   `KD_metrics.C150_target_coverage`，以及 `selected_fraction`。
   - 参照：旧配方的 dev Direct R@10 ≈ 0.27；诊断文档的探针在 direct-only 设置下，SUP ≈ 0.43，KD ≈ 0.47。
   - 这些参照不是保证。≈ 0.27 说明配方没生效；< 0.1 说明崩溃。出现这两种情况都要明确指出。
3. **漂移监控**：查看 C2 训练日志 `$RUN/seed13/NATIVE_C2_SUP/train.attempt_*.jsonl` 和
   `$RUN/seed13/NATIVE_C2_KD/train.attempt_*.jsonl`。报告首行与末行的以下字段：
   `direct_sup_loss`、`evidence_sup_loss`、`direct_kd_loss`、`evidence_kd_loss`、
   `sigma1_R_QT_minus_I`、`sigma2_R_QT_minus_I`、`lr_factor`，以及各 `*_denominator`。
   - `sigma1/sigma2 > 2` 是秩一漂移的信号，要特别标出。
   - `direct_sup_loss` 一直停在 log(候选数) ≈ 6.9 附近，说明学生没在学，也要标出。
4. **证据叙事指标**：取 `$RUN/seed13/eval/{dev,test}/SUMMARY.json` 中的 `narrative` 块和 `contrasts` 块，
   重点是 `evidence_CQET_Real_minus_f0_*`、`KD_minus_SUP_*`、`content_CQET_Real_minus_Swap_implicit`。
   原样贴出数值即可，不要自行解读成“机制成立”。
5. **运行信息**：开始与结束时间、GPU、`git status --short` 的输出；如有失败重启，列出原因和日志路径。

## 5. 可选：消融（只在用户要求时执行）

主实验同时开启了 A、B、C、D，结果无法归因到单项。每个消融是一个独立的 run：

1. 用新的 `RUN` 路径执行第 3 节的第 1 步（`init`）。
2. 在第 2 步（`lock`）**之前**，修改 `$RUN/protocol.json` 中 `student` 下的对应字段。
3. 其余步骤与主实验相同。

每个 run 都会从头重新训练 teacher，单个 run 的耗时与主实验相当。不要在同一台机器上并行跑两个 run，除非已确认内存足够。

| 变体 | 在 `student` 中的修改 |
|---|---|
| A+B（复现探针的配方） | `lr_schedule: "constant"`，`C2.epochs: 1`，`C2.checkpoints: [0, 0.25, 0.5, 0.75, 1]`，`teacher_scored_negatives: false`，`kd_top_k: 0`，`evidence_random_negatives: 0` |
| 去掉余弦衰减 | `lr_schedule: "constant"` |
| 去掉 C1（teacher 打分负样本） | `teacher_scored_negatives: false` |
| 去掉 C2（top-K KD） | `kd_top_k: 0` |
| 去掉 D（证据随机负样本） | `evidence_random_negatives: 0` |

改完后验证 protocol 仍然合法，命令无报错即可：

```bash
conda run -n MMDD python -c "import sys, json; sys.path.insert(0, 'src'); from mmdd_stage1.config import validate_protocol; validate_protocol(json.load(open('$RUN/protocol.json'))); print('ok')"
```
