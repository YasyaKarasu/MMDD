# S2-COL-R1

独立列定位入口为 `src/run_stage2_columns.py`。新增代码在 `mmdd_stage2/column_*.py`，不调用填值、定位、裁剪或 join 验证。普通端到端 gate 不变。

执行目录为 `work/S2-COL-R1`。`SOURCE_AUDIT_BEFORE.json` 记录与上传审计源码的逐文件匹配。主实验使用 canonical train/dev/test；真实可用范围、排除项与自然检索覆盖见 `DATA_AUDIT.json`。监督保存在 `COLUMN_POPULATION.*.jsonl`，reader 只接收 `OBJECTS.jsonl.gz` 中 allowlist 处理过的表格/证据。

先导出经过哈希核验的历史 B13+T0 自然 U，再审计人口：

```bash
conda run -n MMDD python src/prepare_stage2_column_inputs.py --root /home/oycy/MMDD --output /home/oycy/MMDD/work/S2-COL-R1
conda run -n MMDD python src/run_stage2_columns.py --output /home/oycy/MMDD/work/S2-COL-R1 audit \
  --dataset-root /home/oycy/MMDD/output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9 \
  --dataset-root /home/oycy/MMDD/output_wdc_webtable_2000_qwen35_balanced_context_multi_positive_v3 \
  --retrieval /home/oycy/MMDD/work/S2-COL-R1/FROZEN_NATURAL_U.jsonl.gz
```

导出保留已有自然 U 和 T0 在 U 内的排序，证据使用实际 retained_paths。缺失的 test 自然检索可用 `complete_stage2_column_retrieval.py --root /home/oycy/MMDD --output /home/oycy/MMDD/work/S2-COL-R1` 在相同冻结 B13/T0 上补齐；不读取正确 target/列，不新增融合或训练。随后 audit 再传 `--retrieval /home/oycy/MMDD/work/S2-COL-R1/FROZEN_NATURAL_TEST.jsonl.gz`。不将历史锚点声称为最新最强。自然检索缺失整个 query 与已检索 query 的空 E 分开记录。

用户已将最初 A100 要求修正为本机 4090。使用 GPU 1 的 UUID 固定设备，并从 `/tmp` 执行，隔离用户密钥文件：

```bash
cd /tmp
CUDA_VISIBLE_DEVICES=GPU-b1fcd6e2-b542-a1b0-8656-202feee8afc7 \
PYTHONPATH=/home/oycy/MMDD/src \
conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/run_stage2_columns.py \
  --output /home/oycy/MMDD/work/S2-COL-R1 run \
  --model-dir /home/oycy/MMDD/hf_models/Qwen3.5-9B --device cuda:0
```

`run` 顺序为真实可见性探针、两种布局各 32 样本、三臂两个 seed 的 tiny-overfit、两视图正式特征、20 次完整基础样本 epoch、dev 选择、选定 checkpoint 的 O-O/O-R 评测和 dev 扰动诊断。任何 tiny 检查未通过都会停止正式训练。`--tiny-only` 只执行前置检查。完整批次预算为 32；最后不足 32 的批次保留。图像整图统一缩放到最多 262144 pixels，文本总字符预算 12000，目标最多 12 行。预算变更会使缓存失效。

也可用 `cache/train/eval/report` 单独恢复步骤；参数见各子命令 `--help`。C1/C2 共享特征，头部训练在 CPU 上进行。缓存路径由内容指纹寻址，旧 cache 不自动升级。正式训练每个 seed 都要求对应 tiny receipt 通过。

P0 `replay` 单独重放历史 Round1 checkpoint 与其绑定的原 cache。它不是 R25 scorer 在当前 v9 人口的重跑：旧人口 gold 全在显示位置 0，必须与位置基线同时解释。R25 当前人口重放使用新的历史布局前向，按其原列顺序、asset ID 和图像策略执行；状态见单独 P0-R25 receipt。

验证：

```bash
cd /tmp
PYTHONPATH=/home/oycy/MMDD/src conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_stage2_columns.py \
  /home/oycy/MMDD/tests/test_stage2_verifier.py \
  /home/oycy/MMDD/tests/test_stage2_oracle.py \
  /home/oycy/MMDD/tests/test_stage2_reranking.py \
  /home/oycy/MMDD/tests/test_stage2_r12_task_f.py \
  /home/oycy/MMDD/tests/test_stage2_r26.py -q
```

合成测试通过不代表实际 9B 训练完成。以 `RUN_RECEIPTS`、checkpoint hashes、逐 epoch visits、独立标签与完整预测重新计算的指标为证。缺失输出保留分母；无自然检索产物的测试轨不能用于推断证据噪声效应。`RESULTS` 必须保留未执行状态。

完整运行退出成功后，独立核验并生成最终报告：

```bash
cd /tmp
export PYTHONPATH=/home/oycy/MMDD/src
conda run -n MMDD python /home/oycy/MMDD/src/audit_stage2_column_execution.py --output /home/oycy/MMDD/work/S2-COL-R1
conda run -n MMDD python /home/oycy/MMDD/src/plot_stage2_column_trajectories.py --output /home/oycy/MMDD/work/S2-COL-R1
conda run -n MMDD python /home/oycy/MMDD/src/finalize_stage2_columns.py --output /home/oycy/MMDD/work/S2-COL-R1
```

`audit_stage2_column_support.py` 只读取既有 recovery 标注，记录选定 E 的已知行覆盖与未截断正文保留率，不生成新的属性值。最终报告还需要冻结的 24 个 dev 案例审阅记录 `DEV_REVIEW_NOTES.json`；不把自动列名匹配当作外部人工裁决。`finalize_stage2_columns.py` 重算固定模型输入扰动的成对差分与 source-cluster 置信区间，并输出分桶、成本、案例及交付哈希清单。
