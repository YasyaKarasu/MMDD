"""Summarize actual local Student, Teacher and Stage2 costs with separate scopes."""
from __future__ import annotations

import json
from pathlib import Path

from prepare_stage1_r26 import OUT, file_record
from run_stage1_r25 import _json, sha256


def run() -> dict:
    stats = OUT / "statistics"
    barrier = stats / "COST_BENCHMARK_RECEIPT.json"
    if not barrier.exists():
        raise ValueError("Finish the actual benchmark queue before reporting all costs")
    rows = []
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        own = OUT / "rankings" / name / "RETRIEVAL_RECEIPT.json"
        if not own.exists():
            continue  # Verified aliases have no independent measurements.
        retrieval = json.loads(own.read_text())
        index = json.loads((own.parent / "INDEX_RECEIPT.json").read_text())
        measurements = {}
        for module,device in (("student_latency","cpu"),("student_latency","cuda_0"),("teacher_latency","cuda_1")):
            path = stats / module / device / name / "LATENCY_RECEIPT.json"
            receipt = json.loads(path.read_text())
            if sha256(Path(receipt["queries"]["path"])) != receipt["queries"]["sha256"]:
                raise ValueError("Actual timing-query evidence changed")
            if receipt["own_retrieval"]["sha256"] != sha256(own):
                raise ValueError("Cost is not bound to the current own generator")
            measurements[module+"/"+device] = receipt
        teacher = json.loads((OUT / "teacher" / name / "TEACHER_RECEIPT.json").read_text())
        rows.append({"generator":name,"measurements":measurements,"retrieval_cost":retrieval["cost"],
                     "teacher_offline_cache_cost":teacher["cost"],"index_bytes":sum(r["bytes"] for r in index["files"]),
                     "index_built_this_run":index["built_this_run"],"index_load_and_build_seconds":index["load_and_build_seconds"],
                     "objects":index["objects"],"index_receipt":file_record(own.parent / "INDEX_RECEIPT.json"),
                     "checkpoint":file_record(Path(spec["checkpoint"])) if spec["checkpoint"] else None})
    expected = len(json.loads((OUT / "acceptance/raw_rankings/AUDIT.json").read_text())["models"])
    if len(rows) != expected:
        raise ValueError("Missing canonical cost measurements")

    def times(summary: dict) -> str:
        return f"{summary['p50']*1000:.2f} / {summary['p95']*1000:.2f}"

    lines = ["# R26 实测成本","",
        "所有数值来自本地两张RTX4090及本地冻结Qwen feature缓存。每个模型使用固定无标签SHA排序选出的32个dev query，逐query实际调用，不读取结果缓存。两个种子保留为独立行；重复参数的alias不重复计算成本。表中延迟均为p50 / p95，单位ms。共享GPU上小任务可能影响计时。","",
        "## Student检索","",
        "HNSW始终在CPU运行，P/R变换在标注设备运行；计时包含ANN、全部路径、D1与Equal100。cold/warm仅指派生relation vector缓存，feature与索引已在内存。每次重新运行HNSW/路径评分。排除backbone提取、Column、Teacher、exact诊断和索引构建。","",
        "| Canonical generator | CPU cold | CPU warm | GPU0 cold | GPU0 warm | Index GB | Build/load s; built? |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        entries = []
        for device in ("cpu","cuda_0"):
            measurement = row["measurements"]["student_latency/"+device]["latency_seconds"]
            entries += [times(measurement[mode]["total_seconds"]) for mode in ("cold_relation_vectors","warm_relation_vectors")]
        lines.append("| "+row["generator"]+" | "+" | ".join(entries)+f" | {row['index_bytes']/1e9:.3f} | {row['index_load_and_build_seconds']:.2f}; {row['index_built_this_run']} |")
    lines += ["","Build/load列是索引加载与本次构建合计，built=False的数值不能称构建耗时。缓存大小按各模型实际index receipt统计，包含索引与ID/manifest文件；模型权重路径/大小/hash、初始化时间及峰值显存见配套JSON。","",
        "## 固定T0评分","",
        "GPU1逐query重新执行真实QT评分，无pair-score缓存；cold/warm仅指Teacher compression缓存。Equal C18正是Stage2的T0预队列预算。排除Student、backbone、Stage2生成与初始化；本地feature载入时间单独保存，未混入下列算术计时。","",
        "| Canonical generator | Equal100 cold | Equal100 warm | Direct100 warm | Equal18 cold | Equal18 warm | Offline new / cached pairs |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        measurement = row["measurements"]["teacher_latency/cuda_1"]["latency_seconds"]
        entries = [times(measurement[key]) for key in ("Equal_C100/cold_compression","Equal_C100/warm_compression",
            "Direct_C100/warm_compression","Equal_C18/cold_compression","Equal_C18/warm_compression")]
        cache = row["teacher_offline_cache_cost"]
        lines.append("| "+row["generator"]+" | "+" | ".join(entries)+f" | {cache['new_pairs']} / {cache['cached_pairs']} |")
    lines += ["","Offline列来自全U/M评估时的真实pair-cache构建与复用；它受执行顺序影响，不能当模型在线优势。每模型的lookup/hash、实际新pair评分及写库耗时、总请求数均在JSON。全U/M几百pair的离线成本不得冒充固定100在线延迟。CPU T0对应实测也保留在各teacher receipt中。","",
        "## Stage2实际值生成","",
        "以下为固定64query/每generator的值生成成本，所有256/512固定重试计入。模型为本地Qwen3.5-9B；排除localization、列预测、最终semantic verification、初始化及Student/T0。不是端到端Stage2耗时。","",
        "| Generator / condition | Calls | Prompt tokens | Generated tokens | Retries | Failed queries | Generation s total | Query p50 / p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    stage2_path = stats / "stage2/RESULTS.json"
    stage2 = json.loads(stage2_path.read_text())
    for name,generator in stage2["summaries"].items():
        for condition,row in generator["conditions"].items():
            lines.append(f"| {name} / {condition} | {row['generation_calls_including_retries']} | {row['prompt_tokens']} | {row['generated_tokens']} | {row['retry_calls']} | {row['failed_queries']} | {row['generation_seconds']:.2f} | {row['query_generation_seconds_p50']:.2f} / {row['query_generation_seconds_p95']:.2f} |")
    continuation = []
    lines += ["","## 条件Teacher续训与终点评估","",
        "以下直接读取完整10536更新的实际训练receipt。训练秒数含循环、checkpoint及记录开销，排除初始化；同一卡上的三臂并行运行，不能把各臂时长之和当独占GPU小时或总体墙钟。显存为该进程的peak allocated。终点评估为B13/paired-new的U/M全池离线评分及排名保存，按query内目标去重计pair，排除模型初始化；不是C100在线延迟。","",
        "| Teacher | Seed | Updates | Training min | Pair slots | Peak GiB | Offline evaluated pairs | Offline evaluation s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("Tcont","Told","Tnew"):
        for seed in (13,29):
            job = OUT / f"feedback/refinement/{arm}/seed{seed}"
            training_path = job / "TRAINING_RECEIPT.json"
            if not training_path.exists():
                continue
            training = json.loads(training_path.read_text())
            if training["optimizer_updates"] != 10536:
                raise ValueError("Cannot report an incomplete Teacher as a full-budget cost")
            evaluation_path = job / "evaluation/EVALUATION_RECEIPT.json"
            evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
            if evaluation and evaluation["signature"]["training"]["sha256"] != sha256(training_path):
                raise ValueError("Teacher inference cost does not match its actual training endpoint")
            continuation.append({"arm":arm,"seed":seed,"training":training,
                                 "training_source":file_record(training_path),"evaluation":evaluation,
                                 "evaluation_source":file_record(evaluation_path)})
            pairs = str(evaluation["actual_pairs"]) if evaluation else "pending"
            elapsed = f"{evaluation['elapsed_seconds']:.2f}" if evaluation else "pending"
            lines.append(f"| {arm} | {seed} | 10536 | {training['elapsed_seconds']/60:.2f} | {training['pair_slots']} | {training['peak_allocated_bytes']/1024**3:.3f} | {pairs} | {elapsed} |")
    lines += ["","不同模块或不同query分布的分位数不能直接相加作为端到端p50/p95。缺失的条件任务成本保持pending；是否需要再蒸馏由实测门决定，其成本须在真正执行后另行补充。"]
    report_path = stats / "COSTS.md"
    report_path.write_text("\n".join(lines)+"\n")
    result = {"execution_status":"ran","scientific_validity":"valid_for_measured_scopes","full_R26_status":"in_progress",
              "canonical_models":len(rows),"measurements":rows,"stage2":stage2,"teacher_continuation":continuation,
              "barrier":file_record(barrier),"stage2_source":file_record(stage2_path),"report":file_record(report_path),"code":file_record(Path(__file__))}
    _json(stats / "COSTS.json",result)
    return {"canonical_models":len(rows),"report":str(report_path)}


if __name__ == "__main__":
    print(json.dumps(run()))
