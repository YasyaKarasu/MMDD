# R5 Teacher 与文本 span：16 小时补证实验执行书

日期：2026-10-05。执行者：Gemini。目标：保留关系 Teacher，验证其选证据的价值；检验行—属性条件文本定位是否改善正确恢复或成本。**不重训，不追分，不修改现有 R5/R4 产物。**

## 0. 交付目标与边界

必须交付：

1. Teacher 在固定候选池上的 FULL/F0/SWAP/QT_ONLY 排名对照。
2. 固定 R5 候选表与属性计划的 Teacher-vs-cosine 选证据恢复对照。
3. 同一前 3000 字符范围内的原始前缀、lexical span、joint span 对照。
4. 正确恢复抽查、覆盖与成本报告，保留失败、不显著、显式任务损失。

额外实验按预算选择：span entity-only/attribute-only；text-only/image-only。不能为了跑齐列表超过 16 小时。没有收益也是有效结果。

本实验没有验证新训练的 Teacher，没有重新蒸馏，也不把历史暴露的 test 当成 unseen holdout。teacher-ranking 的 F0 是同一个 CQET Teacher 的直接分支；QT_ONLY 才是缓存的独立 QT Teacher。C30 结果条件于 R5 已选池，主要比较用 C150。

Teacher 选证据实验只在**现有最多四条自然 evidence bag 内**各模态选一条，不能声称已证明全湖 evidence retrieval 的最优性。选择阶段不读 gold，不查看 target cell；已选属性仍来自原 R5 selector（该 selector 读过原始 E），因此结论是固定 selector 后的恢复阶段作用，不是彻底去掉 evidence 的全系统消融。

## 1. 输入、代码与输出

固定输入：

- S1：`work/stage1_entitables_r5_s13_sup_rerank`
- S2：`work/stage2_entitables_r5_sup_crop_on`
- 数据：S2/config.json 中的 dataset_root，勿改。
- 模型：本地 `hf_models/Qwen3.5-9B`，冻结。
- frozen cosine embedding：S1/protocol.json 的 `paths.pure_cache_dir/z`，默认自动读取，不重编码。

新增/修改代码：

- `src/run_stage2_experiments.py`：prepare / verify / teacher-ranking / compare / audit-recovery。
- `src/mmdd_stage2/experiments.py`：同源样本、独立实验臂、统计与审查导出。
- `src/mmdd_stage2/text_span.py`：词汇与条件 V-feature span 定位。
- `src/mmdd_stage2/recovery.py`：可选 span 接入；默认无 text_span 配置时保持原始文本输入。
- `src/run_stage2.py`：实验臂执行前验证冻结代码、配置与计划。
- `tests/test_stage2_experiments.py`：CPU 合成测试。

实验代码不得读取 `.env.openai`。所有运行从隔离的 `/tmp` 工作目录发起，只加载本地模型，不调用 OpenAI 或远端 API。不读、搜索、复制或修改该秘密文件。

准备命令要求新输出目录。它只链接只读 catalog、私有复制 matching 缓存；baseline 私有复制对应 recovery。其余臂不复制 recovery，不会错误跳过新推理。不要复制整个旧运行目录来造新实验臂。

每臂的 `EXPERIMENT.json` 记录样本、源文件 SHA256、代码 SHA256、冻结配置/计划 SHA256。`verify` 会拒绝代码或配置变化后的续跑。确需修复代码时记录问题，使用新的输出根，并为受影响的配对臂重新准备；不得改 manifest 掩盖差异。

## 2. 环境与通用命令

在每个执行终端设置一次。以下 `R` 是本次唯一新根；若已存在，换一个新后缀，不覆盖。

```bash
set -euo pipefail
P=/home/oycy/MMDD
S1=$P/work/stage1_entitables_r5_s13_sup_rerank
S2=$P/work/stage2_entitables_r5_sup_crop_on
R=$P/work/r5_teacher_span_16h_20261005
mkdir -p "$R"
TASK_CWD=$(mktemp -d /tmp/mmdd-teacher-span-XXXXXX)
cd "$TASK_CWD"

py() { conda run --no-capture-output -n MMDD python -u "$@"; }
prepare_arm() {
  py "$P/src/run_stage2_experiments.py" prepare \
    --source "$S2" --stage1-run "$S1" --output "$R/$1" \
    --arm "$2" --groups "$3" --splits "${@:4}"
}
run_arm() {
  py "$P/src/run_stage2.py" recover --run-root "$R/$1" --gpu "$2"
  py "$P/src/run_stage2.py" score --run-root "$R/$1"
  py "$P/src/run_stage2.py" evaluate --run-root "$R/$1"
}
score_baseline() {
  py "$P/src/run_stage2.py" score --run-root "$R/$1"
  py "$P/src/run_stage2.py" evaluate --run-root "$R/$1"
}
compare_arms() {
  py "$P/src/run_stage2_experiments.py" compare --method "$R/$1" \
    --reference "$R/$2" --output "$R/comparisons/$1-vs-$2"
}
audit_arm() {
  py "$P/src/run_stage2_experiments.py" audit-recovery \
    --run "$R/$1" --output "$R/audits/$1" --review-groups 60
}
```

准备默认 seed 固定；`--groups N` 按 source group 的稳定 hash 选择，每 split 最多 N 组，整组保留。0 表示全量。它不使用标签或已有成绩，组数不是 query 数。配对各臂必须同 groups/splits。

运行 `run_arm` 时将 stdout/stderr 重定向到对应日志；用两个 tmux 终端分别运行 GPU0/GPU1，每张卡仅一个 GPU 进程。不要在同一臂同时启动两个 recover。`score` 在独立进程使用 CPU MiniLM，首次新文本编码可能有额外时间。

## 3. 0–1 小时：预检与真实模型 smoke

```bash
nvidia-smi
py -m pytest "$P/tests/test_stage2_experiments.py" "$P/tests/test_stage2_bidf.py" -q
py "$P/src/run_stage2_experiments.py" --help
prepare_arm smoke_joint span_joint 2 dev
run_arm smoke_joint 0 > "$R/smoke_joint.log" 2>&1
audit_arm smoke_joint
```

验收：Qwen3.5 的 layers 15/23/27 存在 full-attention `self_attn.v_proj`，forward hook 被触发；row/attribute 标记非空；offset 合法；成功结束后 hook 移除、rope_deltas 恢复；恢复值正常 parse；没有 OOM/输入超限。检查 `recovery/dev/*.json` 中 `text_spans`、`text_localization`。短于 192 token 的 evidence 不调用 localizer；若 smoke 全部短文本，扩大到 8 个 source groups，不能把零 forward 当作模型已验收。

实现者已完成 CPU 测试、真实输入 prepare、真实 tokenizer offset/标记检查；**未运行真实 GPU localizer forward，这一步必须由执行者完成，不能在报告里写成已验证。**

如果真实模型接口不兼容，先保留完整错误及版本，修复代码并补测试；从新根重做 smoke。不得静默退回词汇选段或改 layer 后继续把结果称为预注册 joint。

## 4. Teacher 实验 T1：缓存排名对照（CPU，约 0.5 小时，和 GPU 工作并行）

```bash
py "$P/src/run_stage2_experiments.py" teacher-ranking \
  --stage1-run "$S1" --source "$S2" --output "$R/teacher_ranking" \
  > "$R/teacher_ranking.log" 2>&1
```

使用同一 native_sup C150 候选集：

- FULL：`f0 + 0.5 * (Real.aggregated_score - f0)`，R5 冻结 residual scale。
- F0：同 Teacher 的 Q–T 分数。
- SWAP：同样 0.5 scale 的缓存 Swap 分数。
- QT_ONLY：独立 QT Teacher 缓存。

输出 `METRICS.csv`、`PER_QUERY.csv`、`CONTRASTS.csv`。C150 为主表；`R5_C30_CONDITIONAL` 只作固定入选集合的诊断，存在选择条件，不是无偏的新候选生成比较。

主检验：implicit FULL−SWAP R@10；FULL−F0 用于净证据增量。FULL−QT_ONLY 不能替代它们。预期可能不显著；应保留结果。禁止继续扫描 alpha 或按 test 选择参数。

## 5. Teacher 实验 T2：固定计划的证据选择（全量，两卡各约 3.5–4 小时）

```bash
prepare_arm teacher teacher 0 dev test
prepare_arm cosine cosine 0 dev test
```

两个终端分别执行：

```bash
run_arm teacher 0 > "$R/teacher.log" 2>&1
```

```bash
run_arm cosine 1 > "$R/cosine.log" 2>&1
```

两臂都完成后：

```bash
compare_arms teacher cosine
audit_arm teacher
audit_arm cosine
```

固定：Stage-1 排名、C30、选列结果、donor links、恢复 prompt、图像 crop、匹配与融合。每个原始 view，在每种已存在模态中恰选一条；无该模态就不补，空 bag 保持空。因此两臂每 view 的条数/模态配额相同，但文本长度可能不同，必须报告成本，不能声称 token 完全匹配。

Teacher 按 `raw_QET - f0` 选 E；多个 donor 时均值聚合。cosine 按 frozen Qwen 的整张 Q 与 E 的 cosine 选 E，**不是 Student learned score，也不是 row-conditioned cosine**。相同分数按 ID 排序。原 R5 多证据 bag 只作上下文基线，不能把与它的差异归因于 Teacher，因为预算也变了。

`EVIDENCE_SELECTION.jsonl` 保存原 bag、选中 ID、候选分数。检查 `SELECTION_DIFFERENCES.json`，如两臂选中相同证据的 query 占比高，明确有效干预量；不能仅看大量 ties 宣称等效。

主检验：implicit R@10 与人工核对的正确恢复。支持 Teacher 特殊价值需要真实的正确证据/恢复改善；只有 FULL−QT_ONLY 提升而 T2 无差异，不支持“Teacher 更会选 evidence”。CI 包含 0 记 inconclusive，不记 equal。explicit 损失必须报告。

## 6. 文本实验 S：相同可见范围下的 span（先 pilot，再资源锁定）

### S 的三臂定义

- A baseline：原前 3000 **字符**，使用已有 recovery，零新增 GPU。
- B span_lexical：row cells 与 attribute 的词汇命中，选最多 192 **token** 的连续 span。
- C span_joint：使用冻结 Qwen3.5 后层 V features，行与属性条件共同定位最多 192 token。

B/C 都只看同样的前 3000 字符。按最多 1024 token、重叠 128 token 滑窗；从最高分 token 向较强邻边扩展到 192 token，跨窗口选择 relevance 总和最高的 span。ties 取较早位置。短文本原样保留；全零 map 明确记 `FLAT_PREFIX_SPAN`。特殊控制 token 转义后定位，保存的字符偏移相对于**转义后的可见前缀**；不是 PDF 页码或原始未转义字节偏移。

Joint：evidence 放在条件之前，使 causal V features 的 row/attribute token 可看到 evidence。每层分别求 evidence V 与 row-token mean、attribute-token mean 的 cosine，各自 minmax 后相乘，再跨层平均。层固定 15/23/27。它不是已有图像 RAEA 算法，也没有训练新网络。原始行级 entity gate 仍在完整 3000 字符前缀上运行，不改成 span gate。

图片、已选属性和 evidence IDs 全部固定。这个对照测试恢复阶段的 span，不测试 selector 的文本压缩。

### Pilot：先看运行成本，不看 test 成绩来决定规模

```bash
prepare_arm pilot_baseline baseline 32 dev
prepare_arm pilot_lexical span_lexical 32 dev
prepare_arm pilot_joint span_joint 32 dev
score_baseline pilot_baseline > "$R/pilot_baseline.log" 2>&1
```

两个终端分别跑：

```bash
run_arm pilot_lexical 0 > "$R/pilot_lexical.log" 2>&1
```

```bash
run_arm pilot_joint 1 > "$R/pilot_joint.log" 2>&1
```

```bash
audit_arm pilot_joint
audit_arm pilot_lexical
```

用实际墙钟、`RECOVERY.csv` 的 per-query seconds、span forwards、候选任务量估算确认集成本。定位耗时已包含在 recover seconds，不能重复相加。记录均值和 P95，不直接假设与 query 数严格线性；长文本、多 bag query 会更贵。

**在查看确认集效果前**写 `$R/RESOURCE_LOCK.md`：固定 groups、splits、预计时间、剩余预算及选择依据。从每 split 64/128/256/全量中选能使较慢臂预计 <=3 小时（含 30% 余量）的最大规模；无任何规模能满足时保留 pilot，明确小样本限制。不得根据 C 的指标好坏改变样本。dev/test 均按同一 source hash 规则选，不能挑“有提升”的 query。

下面以每 split 128 组为示例；只允许按资源锁修改 G，不得修改方法超参。

```bash
G=128
prepare_arm span_baseline baseline "$G" dev test
prepare_arm span_lexical span_lexical "$G" dev test
prepare_arm span_joint span_joint "$G" dev test
score_baseline span_baseline > "$R/span_baseline.log" 2>&1
```

两个终端分别跑：

```bash
run_arm span_lexical 0 > "$R/span_lexical.log" 2>&1
```

```bash
run_arm span_joint 1 > "$R/span_joint.log" 2>&1
```

```bash
compare_arms span_joint span_baseline
compare_arms span_lexical span_baseline
compare_arms span_joint span_lexical
audit_arm span_baseline
audit_arm span_lexical
audit_arm span_joint
```

主检验：C−B 的 implicit R@10；C−A、nDCG10、正确恢复及总成本作为预先声明的辅助指标。C>A 但 C≈B 只说明压缩/选段有用，不证明 V-feature 定位的独特价值。正确性相近且**总墙钟/总推理成本**降低可作为效率证据；仅 generation token 减少、却增加大量定位 forward，不能宣称加速。

不要求 CI 为正才“完成实验”。负向、不显著和错失支持 span 都是必须交付的结果。

## 7. 附加实验与 16 小时预算

以两张 4090、每臂全量 recover 约 3.5 小时为基准；以下是墙钟预算，不是把两卡速度直接相加。

| 时间段 | GPU0 | GPU1 | 同期 CPU/人工 |
|---|---|---|---|
| 0–1h | joint smoke | 空闲/故障定位 | 测试、T1、冻结输入 |
| 1–2h | lexical pilot | joint pilot | 成本估算、锁定 S 规模；超时则减组数，不读结果挑样本 |
| 2–6h | T2 teacher 全量 | T2 cosine 全量 | 已有 R5 恢复审计、整理 T1 |
| 6–10h | S lexical 确认集 | S joint 确认集 | 基线评分、人工证据审查 |
| 10–13h | 选一组附加臂 | 选一组附加臂 | 配对统计与负例分析 |
| 13–16h | 不启动新 GPU 任务 | 不启动新 GPU 任务 | 完成审查、对照表、REPORT 与复现清单 |

优先级最高的附加工作是 **E0 恢复正确性审计**（必须做，约 2–3 人工小时，穿插执行）：使用 `audit-recovery` 生成的 MANUAL_REVIEW.jsonl，按 source-group 抽样、每 query 一项，不仅审 VALUE。核对实体、属性、证据支持、值正确性、span 是否保留支持。跨臂使用 query/row/attribute 对齐共同任务；不同臂 singleton 请求集合可能不同，不可直接按 task_id 强配。报告可配对数、无法配对数、抽样分母和审查者；样本不足时只作定性诊断。数据集模型生成的 reference 不是人工真值，不能不复核就当 gold。此任务不能由生成模型自判后标为“人工正确率”。

若没有人类审查者，Gemini 可以先逐项检查证据并填写辅助标注，但必须把 reviewer 标明为 Gemini，称为“模型辅助证据审查”，不能称为人工准确率或独立 gold。人工未复核列保持 unavailable，不因此停止已经授权的计算实验。给用户留下可复核的样本和分歧清单。

GPU 附加组 **E1（优先）：行/属性条件消融**。与 S 相同规模和输入，比较 joint、entity-only、attribute-only。目的：区分联合定位是否真的需要两种条件。可直接运行：

```bash
prepare_arm span_entity span_entity "$G" dev test
prepare_arm span_attribute span_attribute "$G" dev test
# 两个终端，各一条
run_arm span_entity 0 > "$R/span_entity.log" 2>&1
run_arm span_attribute 1 > "$R/span_attribute.log" 2>&1
# 完成后
compare_arms span_joint span_entity
compare_arms span_joint span_attribute
```

成功：joint 在正确实体—属性绑定或下游指标上优于单条件；失败/不确定：联合公式没有额外证据，不能宣称必要。若任何一臂预计超过 3 小时，跳过，不拆分为按完成速度筛出的结果。

GPU 附加组 **E2（替代 E1，不默认两组都跑）：text-only/image-only**。使用相同 G 的 R5 prefix 基线，固定 selector 和计划，只移除 recovery 中另一模态。目的：判断生成阶段模态贡献；不能解释为全 pipeline 去除该模态。

```bash
prepare_arm text_only text_only "$G" dev test
prepare_arm image_only image_only "$G" dev test
# 两个终端，各一条
run_arm text_only 0 > "$R/text_only.log" 2>&1
run_arm image_only 1 > "$R/image_only.log" 2>&1
# 完成后，注意 baseline 是 prefix，不能拿 joint 作唯一参照
compare_arms text_only span_baseline
compare_arms image_only span_baseline
```

成功：加入相应模态带来可核验的新增正确恢复；失败：贡献集中于某一模态，应收窄论文表述。只跑单模态臂而不配同样本 baseline 不可交付。

不安排：KD 重训、alpha 调参、全文 span 扩展、重做所有 selector features、新数据集从零构建。16 小时不够同时可靠完成这些。

## 8. 统计、验收与停止规则

- compare 使用 10000 次 source-group paired bootstrap，seed=20260925，query-weighted mean，点态 percentile 95% CI；输出 delta 用 0–1 单位，报告乘 100 为 pp。
- 两臂必须全量完成预定 population；compare 拒绝缺失/重复 query、不同 source group、不同 Stage-1 排名。fallback 留在总体分母，不能删除。
- 不把“非空 VALUE”“模型输出能解析”“恢复了很多行”当正确性。value_slots 是属性—行槽位；unique_rows_with_value 才是 query 内去重行覆盖。
- 预先列出的多个主检验一起报告；若作总体显著性宣称，应做 Holm 等校正，或明确所有 CI 是未经多重比较校正的探索性结果。不能挑一个有利 K/切片当主结果。
- baseline 旧 recovery 没有新 token counters，缺失不是 0。真实成本比较同时保留进程墙钟与日志，旧基线 tokens 记 unavailable。
- prepare 后不运行 catalog/jobs/features/train-head/plans，它们不是本次受控臂的一部分。
- 每小时在 `$R/STATUS.md` 记录完成项、运行项、耗时和剩余预算。13h 后不启动新的 GPU 臂。
- 接近预算时只提交已完成的**配对实验**；未完成臂标记 incomplete，不用先完成的 query 冒充预定样本结果。

## 9. 最终报告要求

写 `$R/REPORT.zh-CN.md`，并落盘而不是只放会话 artifact。包含：

1. 结论先行：Teacher 特殊价值、文本 span 特殊价值各为 supported / inconclusive / contradicted，及适用范围。
2. 样本与运行身份：每 split queries/source groups、RESOURCE_LOCK、代码 hash、模型版本、GPU、执行命令、实际时间。
3. T1 同池表；T2 Teacher/cosine 表与选 E 改变比例；S 三臂表与 C−B/C−A；已完成附加对照。
4. overall/implicit/explicit 的 R10、NDCG10，以及 delta、CI、W/L/T。R20 作辅助；不省略 explicit 损失。
5. coverage、正确恢复审查、span 保留/丢失支持、失败类型。至少给出正确恢复促成 join、选错属性/实体、span 丢失证据各类可追溯例子，不要求每类一定存在。
6. 耗时、generation tokens（有记录的臂）、localizer forwards、fallback、PARSE_ERROR。区分 recovery 总时间与包含在其中的定位时间。
7. 对论文 claims 的具体建议：哪些可保留，哪些只能作设计，哪些应删除。Teacher 保留不等于所有 Teacher-related claim 自动成立。

必须保留原始日志、EXPERIMENT.json、EVIDENCE_SELECTION.jsonl、recovery、evaluation、comparisons、audits 和人工标注文件。不要提交模型或大数据文件到 git。

## 10. 实现交付时已完成的验证

2026-10-05，全部从 `/tmp` 工作目录执行，未进行 GPU 推理或训练：

```bash
conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_stage2_experiments.py \
  /home/oycy/MMDD/tests/test_stage2_bidf.py -q
```

结果：**39 passed**。覆盖窗口尾部、span 边界与平局、联合条件、字符偏移、短文本、缓存计时、hook 异常清理、默认前缀保持、生成请求去重计费、模态配额、样本整组与嵌套、配对人口拒绝、原目录保护和冻结文件校验。

另已通过：真实 R5 的 Teacher/cosine 全量 prepare（1198 dev + 1166 test）；真实数据单组的 teacher-ranking；独立 baseline 目录的 score/evaluate/audit-recovery/compare；本地 Qwen tokenizer 的 fast offsets 与四个条件标记。预检产物在 `/tmp/mmdd_r5_*preflight*20261005`，仅用于实现验证，不能当论文实验结果，也不要续用为正式实验臂。

真实 GPU V-feature forward、定位质量、正式实验耗时和效果尚未验证；按第 3 节先执行 smoke。
