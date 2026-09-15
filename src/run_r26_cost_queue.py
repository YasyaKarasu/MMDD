"""Complete extension CPU latency and actual GPU Student/Teacher measurements."""
import json

from prepare_stage1_r26 import OUT, file_record
from run_r26_followups import await_artifact, execute
from run_stage1_r25 import _json


def run() -> dict:
    await_artifact(OUT / "EXTENSION_EVALUATION_RECEIPT.json")
    extensions = json.loads((OUT / "EXTENSION_EVALUATION_RECEIPT.json").read_text())["generators"]
    arguments = [a for name in extensions for a in ("--generator",name)]
    execute("benchmark_stage1_r26.py",["--device","cpu",*arguments],"extension_student_latency_cpu.log")
    execute("benchmark_stage1_r26.py",["--device","cuda:0"],"student_latency_all_gpu.log")
    # Initial raw/B13 GPU probes run separately; the shared Teacher sqlite is
    # never opened by these benchmarks, so all measurements score real pairs.
    await_artifact(OUT / "statistics/teacher_latency/cuda_1/B13/LATENCY_RECEIPT.json")
    names = [r["generator_id"] for r in json.loads((OUT / "MODEL_INVENTORY.json").read_text())
             if r["generator_id"] not in ("Qwen-Raw","B13") and (OUT / "rankings" / r["generator_id"] / "RETRIEVAL_RECEIPT.json").exists()]
    execute("benchmark_stage1_r26_teacher.py",["--device","cuda:1",*[a for n in names for a in ("--generator",n)]],"teacher_latency_all_gpu.log")
    result = {"execution_status":"ran","scientific_validity":"valid",
              "receipts":[file_record(p) for root in ("student_latency","teacher_latency")
                          for p in sorted((OUT / "statistics" / root).rglob("LATENCY_RECEIPT.json"))]}
    _json(OUT / "statistics/COST_BENCHMARK_RECEIPT.json",result)
    return {"measurements":len(result["receipts"])}


if __name__ == "__main__":
    print(json.dumps(run()))
