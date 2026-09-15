"""Source-paired, seed-averaged inference for the preregistered score intervention."""
from __future__ import annotations

import csv
import json
from collections import defaultdict, Counter

import numpy as np

from mmdd_stage1.r26_statistics import source_cluster_comparison
from prepare_stage1_r27 import ROOT, OUT, read_json, rows, write_json, record
from prepare_stage2_r27 import write_rows


def analyze() -> dict:
    specs=read_json(ROOT/"mmdd_r26_review/R27_INPUT_LOCK.json")["models"]
    data={}
    metrics=[]
    funnels=[]
    for spec in specs:
        name=spec["generator_id"]
        dest=OUT/"score_handoff"/name
        if not (dest/"EXECUTION.json").exists():
            raise ValueError(f"Incomplete model: {name}")
        data[name]=list(rows(dest/"per_query_metrics.jsonl.gz"))
        baseline=read_json(dest/"BASELINE_REPLAY.json")
        for kind,methods in baseline["metrics"].items():
            for method,values in methods.items():
                for metric,value in values.items():
                    metrics.append({"generator":name,"query_kind":kind,"method":"baseline/"+method,"metric":metric,"value":value})
        for kind in ("overall","implicit","explicit"):
            subset=[r for r in data[name] if kind=="overall" or r["query_kind"]==kind]
            for method,values in subset[0]["metrics"].items():
                for metric in values:
                    metrics.append({"generator":name,"query_kind":kind,"method":method,"metric":metric,"value":float(np.mean([r["metrics"][method][metric] for r in subset]))})
        counts=Counter()
        query_meta={r["query_id"]:r for r in data[name]}
        for row in rows(dest/"eo_funnel.jsonl.gz"):
            for kind in ("overall",query_meta[row["query_id"]]["query_kind"]):
                for endpoint in ("C100","T0_10","E50","Equal50"):
                    a=set(row["A0"]["EO_STRICT"][endpoint]);b=set(row["A1"]["EO_STRICT"][endpoint])
                    for item,value in (("A0",len(a)),("A1",len(b)),("gained",len(b-a)),("lost",len(a-b)),("retained_both",len(a&b))):
                        counts[f"{kind}/{endpoint}/{item}"]+=value
        funnels.append({"generator":name,"strict_pairs":baseline["counts"]["EO_STRICT"],"counts":counts})
        write_json(dest/"diagnostics.json",{"status":"completed","strict_EO_transfers":counts,"negative_control":"QT over U does not add to exact topK when exact topK is contained in U","scientific_scope":"within-model score intervention; cross-model candidate differences do not identify C1 causal root cause"})
    groups={name:[name] for name in data}
    for family in ("R25-C1","R26-O-SUP","R26-E-GRAPH"):
        groups[family+"/two_seed_mean"]=[name for name in data if name.startswith(family+"/")]
    statistics=[]
    for group,names in groups.items():
        indexed=[{r["query_id"]:r for r in data[name]} for name in names]
        assert all(set(x)==set(indexed[0]) for x in indexed)
        for kind in ("overall","implicit","explicit"):
            qids=[q for q,r in indexed[0].items() if kind=="overall" or r["query_kind"]==kind]
            for method,metric in (("T0","recall@10"),("T0","recall@20"),("T0","recall@50"),("E_rank","recall@10"),("C100","raw_recall"),("EO_STRICT","C100"),("EO_STRICT","T0_10")):
                deltas=np.array([np.mean([x[q]["deltas"][method][metric] for x in indexed]) for q in qids])
                sources=[indexed[0][q]["source_table_id"] for q in qids]
                statistics.append({"generator":group,"query_kind":kind,"endpoint":method+"/"+metric,"contrast":"A1-A0","seed_averaging":"per_query_before_source_resampling" if len(names)>1 else "single_frozen_model",**source_cluster_comparison(deltas,sources,replicates=10000,seed=260914)})
    dest=OUT/"statistics";dest.mkdir(parents=True,exist_ok=True)
    with (dest/"metrics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    write_rows(dest/"paired_source_bootstrap.jsonl",statistics)
    write_json(dest/"eo_strict_transfers.json",funnels)
    decisions=[]
    for group in groups:
        main=next(s for s in statistics if s["generator"]==group and s["query_kind"]=="overall" and s["endpoint"]=="T0/recall@10")
        eo=next(s for s in statistics if s["generator"]==group and s["query_kind"]=="overall" and s["endpoint"]=="EO_STRICT/C100")
        verdict="supports_candidate_handoff_only_pending_B" if main["mean_delta"]>0 and eo["mean_delta"]>0 else "does_not_support_current_score_intervention"
        decisions.append({"generator":group,"verdict":verdict,"T0_R10":main,"EO_STRICT_admission":eo})
    write_json(dest/"decision_table.json",decisions)
    return {"models":len(data),"paired_contrasts":len(statistics),"decisions":[{"model":d["generator"],"delta_R10":d["T0_R10"]["mean_delta"],"delta_strict_admission":d["EO_STRICT_admission"]["mean_delta"]} for d in decisions]}


if __name__=="__main__":
    print(json.dumps(analyze()))
