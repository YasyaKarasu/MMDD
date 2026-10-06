# Stage-1 R5：SUP 学生池续训 Teacher → 残差收缩 α → SUP handoff → Stage-2 端到端

给执行者（Gemini）：本文分六部分，**按顺序执行**，每部分通过检查再进入下一部分。
本轮把部署臂从 KD 换成 SUP，并在生成 handoff 之前用一次纯 CPU 的扫描确定证据残差的收缩系数 α。

背景文档：
- R4 端到端报告：`docs/stage1_r4_student_from_r3_teacher_e2e_report.zh-CN.md`
- R3 Teacher 报告：`docs/stage1_r3_teacher_evidence_residual_report.zh-CN.md`
- R4 任务说明（本轮命令格式的来源）：`docs/stage1_r4_student_from_r3_teacher_e2e_gemini_task.zh-CN.md`

---

## 0. 规则（必须遵守）

- 所有命令在仓库根目录 `/home/oycy/MMDD` 下执行，conda 环境 `MMDD`。时间一律 `date -u`。
- 第 3 节的代码补丁**已经合入** `src/mmdd_stage1/export.py`、`src/run_stage1.py` 与 `tests/test_stage1_cqet.py`。
  本轮**不需要**再改这些文件，只按第 3.3 节验证；不要改其它 `src/`、`tests/` 文件。
- 不要读取、查看或修改 `.env.openai`。本实验不需要它。
- 不要修改任何已有的 `work/stage1_r3_*`、`work/stage1_entitables_r4_s13*`、`work/stage1_r4_teacher_sp`、`work/stage2_entitables_r4_*` 目录；它们只读。
- 只跑本文这一个 run 链，不要同时跑其它 run。
- 长任务（续训、评估、export、Stage-2）必须 `setsid nohup ... &` 脱离会话，启动前 `date -u` 记录，启动后把 pid 写入文件。
- 第 4 节的评估会**覆盖**新 run root 里的 `seed13/eval/`；源 run root 不受影响。不要对源 run root 跑第 4 节。
- gate 判定为 FAIL 是研究结论，不是程序错误。照常完成并**如实报告**，不要为了通过而改配置重跑。

---

## 1. 这一轮要回答什么

| # | 问题 | 判定方式 |
|---|---|---|
| Q1 | 在 **SUP** 学生池上续训 Teacher，Teacher 能否在该池上 `Real ≥ 学生 Direct_ANN_R10`？ | `$SP_SUP/eval/dev/SUMMARY.json` 的 `end.Real` vs `native_sup.candidate.Direct_ANN_R10` |
| Q2 | 同一个 Teacher 在 SUP 池上的证据判别力（`Real − Swap`）是否比在 KD 池上强？ | `SUMMARY.json` 的 `end.Real_minus_end.Swap` 置信区间；与 `work/stage1_r4_teacher_sp/eval/dev/SUMMARY.json` 的 KD 池结果对照 |
| Q3 | 证据残差需要收缩多少？α 的 dev 最优值与 test 是否一致？ | 第 3 节的 α 扫描表 |
| Q4 | 换成 SUP handoff 后，Stage-2 端到端 R/nDCG@5,10,15,20 是多少？ | `$S2/evaluation/METRICS.csv` |
| Q5 | 证据的净贡献（`BIDF − VISIBLE_IDF`）在 implicit 上是否比 KD handoff 更大？ | `$S2/evaluation/CONTRASTS.csv`；对照 `work/stage2_entitables_r4_crop_on/evaluation/CONTRASTS.csv` |

**设计要求（必须保持）**：本轮续训的 **起点与 R4 的 KD 池续训完全相同**（`work/stage1_entitables_r4_s13/seed13/TB_CQET/checkpoints/end.pt`），超参数完全相同，**唯一变量是挖掘池（SUP vs KD）**。这样 Q2 的对照才是配对的。

---

## 2. 变量

```bash
cd /home/oycy/MMDD
R4=work/stage1_entitables_r4_s13                    # R4 正式 run（只读）
SPKD=work/stage1_r4_teacher_sp                      # R4 的 KD 池续训产物（只读，对照）
SPKD_RERANK=work/stage1_entitables_r4_s13_sp_rerank # R4 的 KD handoff 评估 run（只读）
SP_SUP=work/stage1_r5_teacher_sp_sup                # 本轮：SUP 池续训 Teacher（新）
RERANK=work/stage1_entitables_r5_s13_sup_rerank     # 本轮：新评估 run root（新）
EXPORT=work/stage1_entitables_r5_s13_sup_rerank/stage2_handoff   # 本轮 handoff（新）
S2=work/stage2_entitables_r5_sup_crop_on            # 本轮 Stage-2（新）
DATASET=output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9
GPU=0
SIDE=1
P=/home/oycy/miniconda3/envs/MMDD/bin/python
S="conda run --no-capture-output -n MMDD python src/run_stage1.py"
S2C="conda run --no-capture-output -n MMDD python src/run_stage2.py"
```

已核实的硬件与路径（`$R4/protocol.json`）：

- main GPU = 物理 0 = `GPU-3d43b1bc-b727-456f-2b9f-e3c3b69eb725`
- side GPU = 物理 1 = `GPU-b1fcd6e2-b542-a1b0-8656-202feee8afc7`
- 起点 Teacher：`$R4/seed13/TB_CQET/checkpoints/end.pt`，sha256 `83a02f639ca090ac290e17d24af738fe9638fc14ee7e65c15365b2d58041463d`，state `d434d423b2c381a6102b77877f16c390c325fafd4e672171b81797ffc9b4487e`
- SUP 学生（dev 选中）：`$R4/seed13/NATIVE_C2_SUP/attempts/attempt_001/checkpoints/snapshot_epoch002.pt`，sha256 `b0d56b217c99aa6a2e9cddfbba79ba82a580f686a6cea9294474130f06b91222`
- 对照（KD 池续训）终点：`$SPKD/checkpoints/end.pt`，sha256 `dd580d081b7f42690ccc7fb803f2ebeb4a284f9fd6fe334da020a2561afc8a0d`，state `2079514dbde5cc064b951e6a9a5166b61abc054a317c471ee5a5baf4d3875e84`

跑之前检查（全部满足才继续）：

```bash
cd /home/oycy/MMDD && date -u
nvidia-smi --query-gpu=index,name,memory.used --format=csv     # 两张 4090 都应接近空闲
free -g                                                         # available ≥ 90 GiB
df -h /home/oycy/MMDD | tail -1                                 # 剩余 ≥ 120 GiB
git status --short                                              # 除 export.py 外应无改动
sha256sum $R4/seed13/TB_CQET/checkpoints/end.pt                # 必须等于 83a02f63...
```

---

## 3. 第一部分的门槛：export 需要一次代码改动（α 收缩）

### 3.1 现状（已核实）

`src/mmdd_stage1/export.py` 的 `_export_split` 直接从冻结评估目录读

- `pools.jsonl.gz`（C30/D150 池）
- `rankings.TB_CQET.Real.jsonl.gz`（target 顺序 + `score`）
- `logits.TB_CQET.Real.jsonl.gz`（每 target 的 `f0`、`raw_QET` 路径分、`aggregated_score`）

`_build_record` 有两条硬约束：`score` 必须与 `aggregated_score` 在 `1e-6` 内相等；`paths` 必须非空。
Stage-2 的 `load_stage1` 还要求 `score` 严格降序。所以**任何对证据残差的缩放必须同时作用于
`ranking["scores"]` 与 `logits[...]["aggregated_score"]`，并重新排序**。

证据残差的定义（`src/mmdd_stage1/models.py:28-32`、`train.py:473`、`losses.py:81`）：

```
path_score(q,e,t) = f0(q,t) + path_head(q,e) + path_head(e,t)
target_score      = f0(q,t) + LME_i( path_head_i 残差 )     该 target 有包
                  = f0(q,t)                                  该 target 无包
```

α 收缩定义为：`target_score(α) = f0 + α · (target_score − f0)`，路径分同理（`raw_QET(α) = f0 + α·(raw_QET − f0)`）。
`α = 1` 等于当前行为；`α = 0` 等于完全不看证据。

### 3.2 补丁（**已在仓库中实现**，你只需要验证）

改动已经落到 `src/mmdd_stage1/export.py` 与 `src/run_stage1.py`，包含：

1. `export.py` 新增 `_apply_residual_scale(ranking, logits, residual_scale)`：对每个有包 target 做
   `aggregated_score = f0 + α·(aggregated_score − f0)`，对每条保留路径做
   `raw_QET = f0 + α·(raw_QET − f0)`，无包 target 保持 `f0` 不变，最后按新分数重排 `target_ids` 与 `scores`。
2. `_export_split(..., residual_scale=1.0)` 在调用 `_build_record` 之前应用它；`_build_record` 同样接收
   `residual_scale`，只用于在 `path_aggregation.table_score` 上追加 `.residual_scale=<α>` 作为追溯标记。
3. `export_stage2(..., residual_scale=1.0)` 校验 `0.0 ≤ α ≤ 2.0`，并把 α 传给 train/dev/test 三处导出。
4. `src/run_stage1.py` 的 `export` 子命令新增 `--residual-scale`（默认 `1.0`）。

`α = 1.0` 走快速返回，逐位等于改动前的行为，因此历史 handoff 不受影响。

**注意一个已核实的细节**：`aggregated_score` 与 `LME(raw_QET)` 在 float32 下存在约 `2.3e-6` 的存储舍入差
（`|agg| ≈ 5–10` 时正好是 float32 的 eps）。这是导出时把 float64 写成 float32 造成的，不是逻辑差异，
对 R@10 的排序没有影响（远小于 k=10 边界上的任何分数间隔）。α 扫描脚本按同一公式重算，结论一致。

### 3.3 跑之前先验证代码已就绪

```bash
cd /home/oycy/MMDD
conda run -n MMDD python -m pytest tests/test_stage1_cqet.py -q          # 必须 64 passed
conda run -n MMDD python -m pytest tests/test_stage2_bidf.py -q          # 必须 25 passed
conda run -n MMDD python src/run_stage1.py export --help | grep -A1 residual-scale
```

对应的单元测试 `test_residual_scale_shrinks_the_evidence_residual_and_keeps_the_ranking_sorted` 已在
`tests/test_stage1_cqet.py` 中，覆盖 `α=1` 逐位不变、`α=0.5` 的分数与路径缩放、无包 target 保持 `f0`、
以及重排后 `scores` 仍严格降序。

**若上述任一命令失败，先修代码再继续；不要跳过第 3 节。** 报告里请贴出三条命令的原始输出。

---

## 4. 第二部分：在 SUP 池上续训 Teacher

Step 1 — 先冒烟（约 10 min）。冒烟写到独立目录，不要占用正式目录：

```bash
date -u
setsid nohup $P src/continue_teacher_on_student_pool.py --run-root $R4 --out-dir $SP_SUP.smoke \
  --gpu $SIDE --student SUP \
  --init-checkpoint $R4/seed13/TB_CQET/checkpoints/end.pt \
  --path-mode pairwise_residual --path-loss-scope bagged \
  --witness-target-weight 0.5 --support-weight 1.0 --support-competitors 16 \
  --search exact --limit 16 > $SP_SUP.smoke.log 2>&1 < /dev/null &
```

确认 `mine` / `train` / `evaluate` 三段都通、`$SP_SUP.smoke/eval/dev/SUMMARY.json` 存在后再进入下一步。
正式跑用另一个目录，冒烟目录保留不动（它的 `IDENTITY.json` 是这次链路的收据）：

```bash
date -u && ls $SP_SUP.smoke/eval/dev/SUMMARY.json
```

Step 2 — 正式续训（预计：挖掘 ~11 min + 训练 ~2.25 h + 评估 ~20 min，共约 3 h）：

```bash
mkdir -p $SP_SUP
date -u
setsid nohup $P src/continue_teacher_on_student_pool.py --run-root $R4 --out-dir $SP_SUP \
  --gpu $SIDE --student SUP \
  --init-checkpoint $R4/seed13/TB_CQET/checkpoints/end.pt \
  --path-mode pairwise_residual --path-loss-scope bagged \
  --witness-target-weight 0.5 --support-weight 1.0 --support-competitors 16 \
  --search exact > $SP_SUP.log 2>&1 < /dev/null &
echo $! > $SP_SUP.pid
```

完成标志：`$SP_SUP/eval/dev/SUMMARY.json` 存在，且 `checkpoints/{init,half,end}.pt` 三个都在。

**立即核对起点没被换掉**（`IDENTITY.json` 的 `init_checkpoint.state_sha256` 必须是 `d434d423...`）：

```bash
python3 -c "
import json
d=json.load(open('$SP_SUP/IDENTITY.json'))
print('student_arm   :', d['student_arm'])                       # 必须是 SUP
print('student_ckpt  :', d['student']['path'])
print('student_state :', d['student']['state_sha256'])           # 必须是 cbdfb8f6...
print('init_state    :', d['init_checkpoint']['state_sha256'])   # 必须是 d434d423...
"
```

**Q1/Q2 的读法**（完成后立即汇报，不要下结论）：

```bash
python3 -c "
import json
d=json.load(open('$SP_SUP/eval/dev/SUMMARY.json'))
for gen in d['generators']:
    v=d['generators'][gen]
    direct=v['candidate']['overall']['Direct_ANN_R10']
    print(f'[{gen}] student Direct_ANN_R10 = {direct:.4f}')
    for p in ('init','half','end'):
        t=v['teacher'][p]
        print(f'  {p:4s} f0={t[\"f0\"][\"overall\"]:.4f} Real={t[\"Real\"][\"overall\"]:.4f} Swap={t[\"Swap\"][\"overall\"]:.4f}'
              f' | implicit f0={t[\"f0\"][\"implicit\"]:.4f} Real={t[\"Real\"][\"implicit\"]:.4f} Swap={t[\"Swap\"][\"implicit\"]:.4f}')
    for k,c in v['contrasts'].items():
        print(f'   {k}: {c[\"mean_delta_pp\"]:+.2f}pp CI={[round(x,2) for x in c[\"ci_95\"]]} WLT={c[\"wlt\"]}')
"
```

同时贴出 KD 池的对照（只读）：

```bash
python3 -c "
import json
d=json.load(open('$SPKD/eval/dev/SUMMARY.json'))
v=d['generators']['native_kd']
for p in ('init','end'):
    t=v['teacher'][p]
    print(f'[KD pool, {p}] overall Real={t[\"Real\"][\"overall\"]:.4f} Swap={t[\"Swap\"][\"overall\"]:.4f}'
          f' | implicit Real={t[\"Real\"][\"implicit\"]:.4f} Swap={t[\"Swap\"][\"implicit\"]:.4f}')
"
```

---

## 5. 第三部分：把新 Teacher 装进评估 run root 并重算冻结指标

### 5.1 复制 R4 的 handoff 评估 run root（已核实可行）

R4 用过的做法：以 `$R4` 为模板复制一份完整 run root，再用 `adopt-teacher --replace` 把续训 Teacher 装进
`seed13/TB_CQET`，然后重跑冻结评估。本轮沿用同一做法，模板换成 `$SPKD_RERANK`（它已含
`ADOPTED_TEACHER_CHAIN.json`、`GLOBAL_SELECTION_FREEZE.json` 与完整的 `seed13/`）。

```bash
date -u
rsync -a --info=progress2 \
  --exclude=stage2_handoff --exclude='eval_*.log' --exclude='export.log' --exclude='alpha_scan.txt' \
  $SPKD_RERANK/ $RERANK/
```

**不要**用 `cp -al`（硬链接）：后续评估会就地覆盖 `seed13/eval/`，硬链接会连带破坏源 run root。

已核实：复制过来的 `protocol.json` 里 `paths.run_root` 字段仍写着源路径也没有关系，所有入口都通过
`resolve_default_paths(protocol_path, run_root)` 用**命令行传入的** `--run-root` 覆盖它。

Step 2 — 安装新 Teacher：

```bash
$S adopt-teacher --run-root $RERANK \
  --ta-dir $SPKD_RERANK/seed13/TA/attempts/attempt_001/checkpoints \
  --tb-dir $SP_SUP/checkpoints \
  --path-mode pairwise_residual \
  --replace \
  --reason "R5: Teacher continued on the SUP student pool, adopted for frozen re-evaluation"
```

说明：`--ta-dir` 沿用 `$SPKD_RERANK` 里已装的 TA（本轮没有重训 TA，Teacher 链的 TA 与 R4 相同）；
`--tb-dir` 指向本轮续训的 `checkpoints/{init,half,end}.pt`。装完确认收据：

```bash
python3 -c "
import json
d=json.load(open('$RERANK/ADOPTED_TEACHER_CHAIN.json'))
print(json.dumps(d['installed']['seed13/TB_CQET'], indent=1))
print('reason:', d['reason'])
"
sha256sum $SP_SUP/checkpoints/end.pt
# 必须与上面 end.pt 的 sha256 一致（应为本轮新产物，不是 dd580d08...）
```

### 5.2 重算 dev 与 test 的冻结评估

`evaluate_split` 是内部函数，没有 CLI 子命令；写一个临时脚本驱动它（下表是已核实的签名）。

```bash
cat > /tmp/reeval_r5.py <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/oycy/MMDD/src")
from mmdd_stage1 import pipeline

run = Path("/home/oycy/MMDD/work/stage1_entitables_r5_s13_sup_rerank")
protocol_path = run / "protocol.json"
split = sys.argv[1]                                   # dev | test
rt = pipeline.load_runtime(protocol_path, run)
freeze = json.loads((run / "GLOBAL_SELECTION_FREEZE.json").read_text(encoding="utf-8"))
print(f"[{split.upper()} EVAL] freeze_sha256={freeze['freeze_sha256']}", flush=True)
result = pipeline.evaluate_split(rt, 13, split, freeze["freeze_sha256"])
print(f"[{split.upper()} EVAL] done", flush=True)
PY
```

两张卡各跑一个 split（**必须是不同 GPU**）：

```bash
date -u
CUDA_VISIBLE_DEVICES=GPU-3d43b1bc-b727-456f-2b9f-e3c3b69eb725 \
  setsid nohup $P /tmp/reeval_r5.py dev > $RERANK/eval_dev.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=GPU-b1fcd6e2-b542-a1b0-8656-202feee8afc7 \
  setsid nohup $P /tmp/reeval_r5.py test > $RERANK/eval_test.log 2>&1 < /dev/null &
```

预计每个 split 约 **45–50 min**（R4 实测 dev 2854.8 s / test 2694.9 s）。
两次都要跑完（`[DEV EVAL] done` / `[TEST EVAL] done`）才进入下一节；日志缺失就是没跑完。

跑完后重写汇总（`_reports` 会写 `reports/RESULTS.md` 与 `reports/DECISION.json`）并落 `FINAL_EVALUATION.json`：

```bash
python3 -c "
import json, sys
from pathlib import Path
sys.path.insert(0,'/home/oycy/MMDD/src')
from mmdd_stage1 import pipeline
run=Path('$RERANK')
rt=pipeline.load_runtime(run/'protocol.json', run)
freeze=json.loads((run/'GLOBAL_SELECTION_FREEZE.json').read_text())
res={13:{'seed':13,'global_freeze_sha256':freeze['freeze_sha256'],'splits':{
    s: json.loads((run/f'seed13/eval/{s}/SUMMARY.json').read_text()) for s in ('dev','test')}}}
(run/'seed13/FINAL_EVALUATION.json').write_text(json.dumps(res[13], indent=2))
pipeline._reports(rt, res)
print(open(run/'reports/RESULTS.md').read().splitlines()[:5])
"
```

**核对**：`$RERANK/seed13/eval/dev/native_sup/` 里 `rankings.TB_CQET.Real.jsonl.gz` 的
`teacher_state_hash` 必须等于 `$SP_SUP/checkpoints/end.pt` 的 state hash（不再是 `2079514d...`）：

```bash
python3 -c "
import gzip,json
p='$RERANK/seed13/eval/dev/native_sup/rankings.TB_CQET.Real.jsonl.gz'
with gzip.open(p,'rt') as fh: r=json.loads(fh.readline())
print('teacher_state_hash:', r['teacher_state_hash'])
"
```

**可比性检查（必须做，决定后面的 α 与 Stage-2 对比是否干净）**：新 run root 里重建的学生池
必须与 `$SPKD_RERANK` 的逐位一致（同一个学生、同一份特征、同一 seed，只有 Teacher 应该不同）。

```bash
python3 -c "
import json
for r in ('stage1_entitables_r4_s13_sp_rerank','stage1_entitables_r5_s13_sup_rerank'):
    for gen in ('native_sup',):
        d=json.load(open(f'work/{r}/seed13/eval/dev/{gen}/POOL_MANIFEST.json'))
        print(f'{r:42s} {gen} pool_id={d.get("pool_id")} teacher={d.get("teacher_state_hash","")[:12]}')
"
```

期望：两个 `pool_id` **相同**，`teacher_state_hash` **不同**。若 `pool_id` 不同，说明学生池被重建得不一致，
必须在汇报里写明，并在 Stage-2 对比时把这一点作为混淆项列出。

---

## 6. 第四部分：α 扫描（纯 CPU，不改任何权重）

**在生成 handoff 之前**在 dev 上选 α。输入是本轮重算后的 SUP 池 logits。

```bash
cat > /tmp/alpha_scan_r5.py <<'PY'
"""Pick the residual shrinkage alpha on dev; test is only a regression check."""
import gzip, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/oycy/MMDD/src")
from pathlib import Path
from mmdd_dataset.wdc_runtime import iter_dataset_artifact

RUN = Path("/home/oycy/MMDD/work/stage1_entitables_r5_s13_sup_rerank")
DATASET = Path("/home/oycy/MMDD/output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9")
ARM = "native_sup"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

gold, reasons = defaultdict(set), defaultdict(set)
for row in iter_dataset_artifact(DATASET, "qrels"):
    if row.get("split") in ("dev", "test") and float(row.get("rel", 0)) > 0:
        gold[str(row["query_table_id"])].add(str(row["target_table_id"]))
        reasons[str(row["query_table_id"])].add(row["reason"])

def kind(q):
    r = reasons[q]
    if r == {"model_recoverable_join_column"}: return "implicit"
    if r == {"explicit_visible_join_column"}: return "explicit"
    return "mixed"

def lme(values):
    a = np.asarray(values, dtype=np.float64); m = a.max()
    return float(m + np.log(np.mean(np.exp(a - m))))

def load(split):
    data = defaultdict(list)
    p = RUN / f"seed13/eval/{split}/{ARM}/logits.TB_CQET.Real.jsonl.gz"
    with gzip.open(p, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            data[r["query_id"]].append(
                (r["target_id"], float(r["f0"]), [float(x["raw_QET"]) for x in r["paths"]])
            )
    return data

def order_at(data, alpha):
    out = {}
    for q, rows in data.items():
        scored = []
        for target, f0, paths in rows:
            score = f0 + alpha * (lme(paths) - f0) if paths else f0
            scored.append((score, target))
        scored.sort(key=lambda z: (-z[0], z[1]))
        out[q] = [t for _, t in scored]
    return out

def recall(order, segment):
    v = []
    for q, o in order.items():
        g = gold.get(q)
        if not g or (segment != "overall" and kind(q) != segment):
            continue
        v.append(len(set(o[:10]) & g) / len(g))
    return float(np.mean(v) * 100)

for split in ("dev", "test"):
    data = load(split)
    print(f"\n##### {split}  ({len(data)} queries)")
    print(f"{'alpha':>6} {'overall':>8} {'implicit':>9} {'explicit':>9}  {'min_all':>8}")
    for a in ALPHAS:
        o = order_at(data, a)
        ov, im, ex = recall(o, "overall"), recall(o, "implicit"), recall(o, "explicit")
        print(f"{a:6.2f} {ov:8.2f} {im:9.2f} {ex:9.2f}  {min(ov,im,ex):8.2f}")
PY
python3 /tmp/alpha_scan_r5.py | tee $RERANK/alpha_scan.txt
```

**选 α 的规则（写进报告，不要事后改）**：

1. 在 `α ∈ {0, 0.25, 0.5, 0.75, 1.0}` 上，取 **dev implicit R@10 最高**的点；
2. 若该点的 **dev overall R@10 低于 α=0**，则退到满足该约束的最大 α，并在报告中说明；
3. test 只作回归确认，**不参与选择**。

参照（R4 KD handoff 上按同一规则算出的结果，供你判断本轮方向是否一致）：
`α=0.5` 在 dev/test 上都优于 `α=1`（KD 池：dev overall 50.11 vs 49.86、dev implicit 45.97 vs 45.30；
test overall 50.13 vs 48.67、test implicit 46.74 vs 45.54）。

**必须汇报**：完整的 α 表（两个 split、三个 population）、被选中的 α、以及"若改用 α=1 会差多少"。
如果 dev 与 test 的最优 α 不一致，**如实报告**，不要为了好看挑一个。

---

## 7. 第五部分：导出 SUP handoff

```bash
date -u
export ALPHA=0.5          # ← 换成第 6 节选出的值
setsid nohup $S export --run-root $RERANK --output-dir $EXPORT \
  --arm SUP --top-k 50 --evidence-path-k 4 --residual-scale $ALPHA \
  > $RERANK/export.log 2>&1 < /dev/null &
echo $! > $RERANK/export.pid
```

预计约 **1.3 h**（train 需现算，dev/test 复用冻结评估）。

完成后核对（写进汇报）：

```bash
python3 -c "
import json
g=json.load(open('$EXPORT/stage1_gate.json'))
print('completed_stage :', g['completed_stage'])
print('arm             :', g['stage1_selection']['arm'])          # 必须是 NATIVE_C2_SUP
print('selected_fraction:', g['stage1_selection']['selected_fraction'])
print('dev_evidence_coverage:', json.dumps(g['best_metrics']))
"
python3 -c "
import json
m=json.load(open('$EXPORT/EXPORT_MANIFEST.json'))
for split,v in m['retrievals'].items(): print(split, v['records'], v['sha256'][:16])
"
wc -l $EXPORT/retrieval.train.jsonl $EXPORT/retrieval.dev.jsonl $EXPORT/retrieval.test.jsonl
# 期望 12630 / 1198 / 1166
```

若 `adopt-teacher` 或 `export` 报 "completed source differs"，按提示执行：

```bash
$S amend-source --run-root $RERANK --amendment-id r5-export-residual-scale \
  --carry TA TB_CQET --reason "R5 export adds the residual-scale handoff option"
```

---

## 8. 第六部分：Stage-2 端到端重训与评测

与 R4 完全同配方（crop-on 主线），只换 handoff。

```bash
date -u
$S2C init --run-root $S2 --stage1-handoff $EXPORT
$S2C catalog --run-root $S2                      # CPU
$S2C jobs    --run-root $S2                      # CPU
$S2C features --run-root $S2 --split train --gpu $SIDE
$S2C features --run-root $S2 --split dev   --gpu $SIDE
$S2C features --run-root $S2 --split test  --gpu $SIDE
$S2C train-head --run-root $S2                   # CPU，20 epoch
$S2C plans      --run-root $S2                   # CPU
$S2C recover    --run-root $S2 --gpu $SIDE       # GPU，9B 行级恢复，最耗时
$S2C score      --run-root $S2                   # CPU
$S2C evaluate   --run-root $S2                   # CPU
```

- 全部串行，**不要**在跑 `recover` 时并行其它 9B 任务；
- `features` 与 `recover` 可断点续跑（已有文件会跳过）；
- 预计总耗时约 **3.5–4 h**（R4 实测 13:56 → 17:30）；
- 长任务用 `setsid nohup ... &` 并记录 pid（`catalog`/`jobs`/`train-head`/`plans`/`score`/`evaluate` 是 CPU 短任务，可前台跑）。

**Q4/Q5 的读法**：

```bash
python3 - <<'PY'
import csv
from pathlib import Path
for run in ("work/stage2_entitables_r5_sup_crop_on", "work/stage2_entitables_r4_crop_on"):
    p = Path(run) / "evaluation/METRICS.csv"
    if not p.exists():
        print(run, "missing"); continue
    print("=====", run)
    for r in csv.DictReader(p.open()):
        if r["policy"] in ("STAGE1", "VISIBLE_IDF_RRF60", "BRIDGE_RRF60", "BIDF_RRF60"):
            print(f"  {r['split']:5s} {r['population']:9s} {r['policy']:18s} n={r['queries']:>5s} "
                  f"R10={float(r['R10'])*100:6.2f} NDCG10={float(r['NDCG10'])*100:6.2f} R20={float(r['R20'])*100:6.2f}")
PY
```

关键对照（写进汇报，与 R4 的同名行并排）：

```bash
python3 - <<'PY'
import csv
from pathlib import Path
for run in ("work/stage2_entitables_r5_sup_crop_on", "work/stage2_entitables_r4_crop_on"):
    p = Path(run) / "evaluation/CONTRASTS.csv"
    print("=====", run)
    for r in csv.DictReader(p.open()):
        if r["method"] == "BIDF_RRF60" and r["reference"] == "VISIBLE_IDF_RRF60" and r["metric"] in ("R10", "NDCG10"):
            print(f"  {r['split']:5s} {r['population']:9s} {r['metric']:7s} delta={float(r['delta'])*100:+.2f}pp "
                  f"CI=[{float(r['CI_low'])*100:+.2f},{float(r['CI_high'])*100:+.2f}] W/L/T={r['W']}/{r['L']}/{r['T']}")
PY
```

---

## 9. 这个任务不做的事

- 不重蒸 KD 学生，不改 KD 的损失配方；
- 不改 Stage-1 的 `path_mode`、损失、协议模板或任何 gate 阈值；
- 不改 Stage-2 的 `candidate_scope`、`rrf_constant`、`matching.tau_cosine`、`crop.*` 或恢复输出上限；
- 不跑 crop-off 臂（R4 已证明无差异）；
- 不用 query 标签（implicit/explicit）做路由或选样；
- 不为了让指标好看而挑选样例、调阈值或补选数据；
- 不在同一张 GPU 上并行两个 9B 恢复任务。

---

## 10. 汇报模板（每部分完成后）

```
[部分 N] 状态：COMPLETE / FAILED / PARTIAL
命令：(原样)
时间：(UTC 起止，wall)
产物：路径 + 关键文件 sha256
关键数字：(只列原始数值，不下结论)
异常：(报错原文、日志路径；没有就写 none)
```

全部完成后，另外交一份汇总：

1. Q1/Q2：SUP 池 `end.Real` vs 学生 Direct；`Real − Swap` 的 CI；与 KD 池并排。
2. Q3：α 扫描表 + 选中的 α + test 是否一致；若不一致写清楚。
3. Q4/Q5：`METRICS.csv` / `CONTRASTS.csv` 的关键行，与 R4 crop-on 并排。
4. 与 R4 的差异归因：哪些变化来自换池、哪些来自 α、哪些来自 handoff 重算。**只写数据支持的结论。**
