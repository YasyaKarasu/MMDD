"""Frozen-corpus text-length audit and evidence-only prompt counterfactuals.

Run from an isolated working directory. No client configuration, training, or
historical-case selection is used. Each phase writes only to --output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from cache_stage1_features import _load_embedder_class, encode_preprocessed_inputs
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore


BINS = ("0-2", "3-5", "6-10", "11-20", "21+")
PLACEHOLDER = re.compile(r"\[(?:url|image|img|missing|null|none|unk|unknown|n/?a)\]", re.I)
ARMS = ("original", "minimal")


def bucket(n: int) -> str:
    """Disjoint payload-token bins (20 belongs to 11-20)."""
    return next((label for limit, label in zip((2, 5, 10, 20), BINS) if n <= limit), BINS[-1])


def rows(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        for line in f:
            yield json.loads(line)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path: Path, values) -> None:
    with (gzip.open(path, "wt") if path.suffix == ".gz" else path.open("w")) as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def summary(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"n": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            **{name: float(np.quantile(a, q)) for name, q in (("p10", .1), ("median", .5), ("p90", .9))}}


def sources(root: Path) -> dict[str, Path]:
    r10 = root / "work/stage1_optimization_r10_20260907"
    r26 = root / "work/stage1_optimization_r26_20260914"
    return {"objects": r10 / "stage1_data/stage1_objects.jsonl",
            "corpus": r10 / "stage1_data/stage1_corpus.jsonl",
            "features": r10 / "features_qwen3_vl_embedding_8b",
            "queries": r26 / "common/dev_queries.jsonl",
            "B13": r26 / "rankings/B13/rankings.jsonl.gz",
            "Qwen-Raw": r26 / "rankings/Qwen-Raw/rankings.jsonl.gz",
            "student": r26 / "recovered/B13/step_000178.pt",
            "model": root / "hf_models/Qwen3-VL-Embedding-8B"}


def prepare(root: Path, output: Path, seed: int, per_bin: int) -> None:
    from transformers import AutoTokenizer

    ps = sources(root)
    corpus = {r["object_id"] for r in rows(ps["corpus"])}
    tokenizer = AutoTokenizer.from_pretrained(ps["model"], local_files_only=True)
    evidence = {}
    for r in rows(ps["objects"]):
        if r["object_type"] != "text" or r["object_id"] not in corpus:
            continue
        text = r["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        removed = PLACEHOLDER.sub("", text)
        placeholder_tokens = sum(len(tokenizer.encode(m.group(), add_special_tokens=False)) for m in PLACEHOLDER.finditer(text))
        evidence[r["object_id"]] = {"object_id": r["object_id"], "text": text,
            "tokens": len(ids), "bucket": bucket(len(ids)), "placeholder_count": len(PLACEHOLDER.findall(text)),
            "placeholder_token_fraction": min(1., placeholder_tokens / max(1, len(ids))),
            "remaining_tokens": len(tokenizer.encode(removed, add_special_tokens=False)) if removed != text else len(ids)}
    print(json.dumps({"event": "inventory", "texts": len(evidence), "bins": Counter(r["bucket"] for r in evidence.values())}), flush=True)
    metrics = {}
    query_ids = None
    occurrences = Counter()
    strict_evidence = set()
    for generator in ("B13", "Qwen-Raw"):
        groups = {b: defaultdict(list) for b in BINS}
        seen_e = {b: set() for b in BINS}
        counts = {b: Counter() for b in BINS}
        qids = []
        strict_pair_count = 0
        slim = []
        for row in rows(ps[generator]):
            q = row["query_id"]
            qids.append(q)
            truth = set(row["positive_target_ids"])
            strict = truth & (set(row["E_target_ids"]) - set(row["D100_EXACT"]) - set(row["rankings"]["D100_ANN"]))
            strict_pair_count += len(strict)
            qe = {}
            targets = {r["target_id"]: r for r in row["E_paths"]}
            for target in row["E_pre_retention"]:
                tid = target["target_id"]
                retained = set(targets.get(tid, {}).get("selected_evidence_ids", []))
                for p in target["paths"]:
                    if p.get("evidence_type") != "text":
                        continue
                    eid = p["evidence_id"]
                    b = evidence[eid]["bucket"]
                    qe[eid] = p["query_evidence_score"]
                    groups[b]["retrieved_et"].append(p["evidence_target_score"])
                    groups[b]["retrieved_path"].append(p["path_score"])
                    counts[b]["paths"] += 1
                    counts[b]["retained_paths"] += eid in retained
                    counts[b]["strict_retained_paths"] += tid in strict and eid in retained
                    if tid in strict and eid in retained:
                        strict_evidence.add(eid)
                    slim.append({"query_id": q, "evidence_id": eid, "target_id": tid,
                        "bucket": b, "qe": p["query_evidence_score"], "et": p["evidence_target_score"],
                        "path": p["path_score"], "retained": eid in retained,
                        "strict_target": tid in strict, "qrel_target": tid in truth})
            assert len(qe) == 20, (generator, q, len(qe))
            for eid, score in qe.items():
                b = evidence[eid]["bucket"]
                groups[b]["retrieved_qe"].append(score)
                counts[b]["qe_top20_occurrences"] += 1
                seen_e[b].add(eid)
                occurrences[(generator, eid)] += 1
        if query_ids is not None:
            assert query_ids == qids
        query_ids = qids
        total_strict = sum(c["strict_retained_paths"] for c in counts.values())
        metrics[generator] = {"queries": len(qids), "strict_target_pairs": strict_pair_count, "bins": {}}
        for b in BINS:
            n = sum(e["bucket"] == b for e in evidence.values())
            c = counts[b]
            metrics[generator]["bins"][b] = {"corpus_n": n, **c,
                "unique_retrieved_evidence": len(seen_e[b]),
                "top20_slot_share": c["qe_top20_occurrences"] / (len(qids)*20),
                "top20_rate_per_query_evidence_opportunity": c["qe_top20_occurrences"] / (len(qids)*n) if n else None,
                "ever_retrieved_fraction": len(seen_e[b])/n if n else None,
                "strict_witness_share": c["strict_retained_paths"]/total_strict if total_strict else None,
                "strict_fraction_of_retained_paths": c["strict_retained_paths"]/c["retained_paths"] if c["retained_paths"] else None,
                **{name: summary(values) for name, values in groups[b].items()}}
        write_rows(output / f"paths_{generator}.jsonl.gz", slim)
        print(json.dumps({"event": "archived_paths", "generator": generator, "queries": len(qids), "paths": len(slim)}), flush=True)
    rng = np.random.default_rng(seed)
    reasons = defaultdict(list)
    for b in BINS:
        ids = sorted(e for e in evidence if evidence[e]["bucket"] == b)
        for e in rng.choice(ids, size=min(per_bin, len(ids)), replace=False).tolist():
            reasons[e].append("random_bin")
        for g in ("B13", "Qwen-Raw"):
            for e in sorted(ids, key=lambda e: (-occurrences[g, e], e))[:12]:
                if occurrences[g, e]:
                    reasons[e].append(f"hub_{g}")
    for e in sorted(strict_evidence, key=lambda e: hashlib.sha256(f"{seed}/{e}".encode()).hexdigest())[:48]:
        reasons[e].append("strict_witness")
    placeholder_ids = sorted((e for e in evidence if evidence[e]["placeholder_count"]),
                             key=lambda e: (-evidence[e]["placeholder_token_fraction"], e))
    for e in placeholder_ids[:48]:
        reasons[e].append("placeholder_dense")
    sample = [{**evidence[e], "selection": why, "top20_counts": {g: occurrences[g,e] for g in ("B13", "Qwen-Raw")}}
              for e, why in sorted(reasons.items())]
    write_rows(output / "evidence_inventory.jsonl.gz", evidence.values())
    write_rows(output / "sample.jsonl", sample)
    write_json(output / "baseline_buckets.json", metrics)
    write_json(output / "inventory.json", {"texts": len(evidence), "bins": Counter(e["bucket"] for e in evidence.values()),
        "placeholder_texts": len(placeholder_ids), "placeholder_ge50pct": sum(e["placeholder_token_fraction"] >= .5 for e in evidence.values()),
        "sample_size": len(sample), "sample_bins": Counter(e["bucket"] for e in sample), "seed": seed, "random_per_bin": per_bin})
    write_json(output / "PROTOCOL.json", {"inputs": {k: fingerprint(v) for k, v in ps.items() if v.is_file()},
        "feature_metadata": fingerprint(ps["features"] / "metadata.json"),
        "feature_manifest": fingerprint(ps["features"] / "manifest.jsonl"),
        "wrapper": fingerprint(ps["model"] / "scripts/qwen3_vl_embedding.py"),
        "tokenizer": fingerprint(ps["model"] / "tokenizer.json"),
        "seed": seed, "bins": BINS, "selection": "seeded random per nonempty bin plus current-corpus hubs and strict witnesses; no historical cases",
        "strict_definition": "G intersect E minus union(D_ANN100,D_EXACT100)",
        "topk_definition": "text Q->E top20; path occurrences and unique Q/E occurrences counted separately",
        "witness_definition": "retained text path to a strict-EO qrel target; not evidence-semantic truth",
        "counterfactual": "E-only intervention; Q/T and every competing corpus embedding remain frozen; exact text-corpus rank, not end-to-end retrieval",
        "placeholder_pattern": PLACEHOLDER.pattern, "arms": ARMS,
        "training": False, "configured_client": False})


def render_inputs(embedder, texts: list[str], arm: str, instruction: str):
    """Avoid wrapper's instruction and empty-payload fallbacks in interventions."""
    conversations = []
    for text in texts:
        if arm == "original" or arm == "wrapper_empty":
            conversation = embedder.format_model_input(text=text, instruction=instruction)
        else:
            content = [{"type": "text", "text": text}]
            conversation = [{"role": "user", "content": content}]
            prompt = "Represent this text for retrieval." if arm in ("minimal", "minimal_empty") else instruction
            conversation.insert(0, {"role": "system", "content": [{"type": "text", "text": prompt}]})
        conversations.append(conversation)
    return embedder._preprocess_inputs(conversations)


def encode(root: Path, output: Path, device: str, batch_size: int) -> None:
    ps = sources(root)
    torch.cuda.set_device(device)
    cls = _load_embedder_class(ps["model"])
    embedder = cls(model_name_or_path=str(ps["model"]), torch_dtype=torch.bfloat16,
                   attn_implementation="sdpa", local_files_only=True)
    instruction = json.loads((ps["features"] / "metadata.json").read_text())["embedding_instructions"]["evidence_text"]
    sample = list(rows(output / "sample.jsonl"))
    jobs = [(r["object_id"], arm, r["text"]) for r in sample for arm in ARMS]
    jobs += [("__prompt__", "empty", ""), ("__prompt__", "wrapper_empty", ""), ("__prompt__", "minimal_empty", "")]
    jobs.sort(key=lambda x: (x[1], len(x[2])))
    vectors, input_lengths = {}, {}
    started = time.monotonic()
    for arm in (*ARMS, "empty", "wrapper_empty", "minimal_empty"):
        arm_jobs = [j for j in jobs if j[1] == arm]
        for start in range(0, len(arm_jobs), batch_size):
            batch = arm_jobs[start:start+batch_size]
            inputs = render_inputs(embedder, [j[2] for j in batch], arm, instruction)
            encoded = encode_preprocessed_inputs(embedder, inputs, include_hidden=False)
            for job, (vector, _, _), n in zip(batch, encoded, inputs["attention_mask"].sum(1).tolist()):
                vectors[job[0]+"/"+arm] = vector
                input_lengths[job[0]+"/"+arm] = n
            if start % (batch_size*8) == 0:
                print(json.dumps({"event": "encode", "arm": arm, "done": start+len(batch), "total": len(arm_jobs),
                                  "seconds": time.monotonic()-started}), flush=True)
    torch.save({"vectors": vectors, "input_lengths": input_lengths}, output / "counterfactual_embeddings.pt")
    write_json(output / "ENCODING.json", {"seconds": time.monotonic()-started, "sequences": len(vectors),
        "model": str(ps["model"]), "dtype": "bfloat16", "attention": "sdpa", "max_length": embedder.max_length,
        "device": torch.cuda.get_device_name(), "empty_payload": "explicit empty user text, bypass NULL fallback",
        "wrapper_empty": "official wrapper substitutes NULL; measured separately",
        "template": "same system/user chat template and final-token pooling in both prompt arms",
        "minimal": "Represent this text for retrieval.", "original": instruction})


@torch.inference_mode()
def analyze(root: Path, output: Path, device: str) -> None:
    ps = sources(root)
    torch.cuda.set_device(device)
    inventory = list(rows(output / "evidence_inventory.jsonl.gz"))
    sample = list(rows(output / "sample.jsonl"))
    eids = [r["object_id"] for r in inventory]
    epos = {e: i for i, e in enumerate(eids)}
    sids = [r["object_id"] for r in sample]
    spos = {e: i for i, e in enumerate(sids)}
    qids = [r["query_id"] for r in rows(ps["queries"])]
    qpos = {q: i for i, q in enumerate(qids)}
    store = FeatureStore.from_path(ps["features"], cache_size=512)
    print(json.dumps({"event": "load_cached_embeddings", "texts": len(eids), "queries": len(qids)}), flush=True)
    ee = torch.stack([store.embedding_features(e).embedding for e in eids]).to(device)
    qq = torch.stack([store.embedding_features(q).embedding for q in qids]).to(device)
    encoded = torch.load(output / "counterfactual_embeddings.pt", weights_only=True)
    vectors = encoded["vectors"]
    cf = {arm: torch.stack([vectors.get(e+"/"+arm, vectors[e+"/original"]) for e in sids]).to(device) for arm in ARMS}
    original_cached = ee[[epos[e] for e in sids]]
    cos_empty = (cf["original"] @ vectors["__prompt__/empty"].to(device)).cpu().numpy()
    cos_null = (cf["original"] @ vectors["__prompt__/wrapper_empty"].to(device)).cpu().numpy()
    cos_minimal_empty = (cf["minimal"] @ vectors["__prompt__/minimal_empty"].to(device)).cpu().numpy()
    parity = (cf["original"] * original_cached).sum(1).cpu().numpy()
    details = [{**r, "cos_full_empty": float(cos_empty[i]), "cos_full_wrapper_null": float(cos_null[i]),
                "cos_minimal_empty": float(cos_minimal_empty[i]), "cos_reencode_cache": float(parity[i]),
                "input_tokens": {arm: encoded["input_lengths"].get(r["object_id"]+"/"+arm) for arm in ARMS}}
               for i, r in enumerate(sample)]
    write_rows(output / "embedding_diagnostics.jsonl", details)
    embedding_stats = {}
    for scope in ("random_bin", "all_selected"):
        embedding_stats[scope] = {}
        for b in BINS:
            ix = [i for i, r in enumerate(sample) if r["bucket"] == b and (scope == "all_selected" or scope in r["selection"])]
            embedding_stats[scope][b] = {"cos_full_empty": summary(cos_empty[ix]), "cos_full_wrapper_null": summary(cos_null[ix]),
                                        "cos_minimal_empty": summary(cos_minimal_empty[ix]),
                                        "reencode_cache": summary(parity[ix])}
    write_json(output / "embedding_summary.json", embedding_stats)
    if float(parity.min()) < .99:
        raise ValueError("Original-prompt re-encoding does not match cache; inspect before interpreting counterfactuals")
    print(json.dumps({"event": "embedding_parity", "min_cos": float(parity.min()), "mean_cos": float(parity.mean())}), flush=True)
    student = load_student(ps["student"], torch.device(device)).eval()
    bins_tensor = torch.tensor([BINS.index(r["bucket"]) for r in inventory], device=device)
    baseline_all = {}
    qe_matrices = {}
    qe_reports = {}
    for generator, model in (("Qwen-Raw", None), ("B13", student)):
        qvec = qq if model is None else model.relation_query(qq, "table", "text")
        evec = ee if model is None else model.index_vector(ee, "text")
        scores = qvec @ evec.T
        cached_sample_qe = scores[:, [epos[e] for e in sids]]
        top_positions = scores.topk(20, dim=1).indices
        baseline_all[generator] = {}
        for bi, b in enumerate(BINS):
            mask = bins_tensor == bi
            n = int(mask.sum())
            selected_n = int((bins_tensor[top_positions] == bi).sum())
            part = scores[:, mask]
            baseline_all[generator][b] = {"corpus_n": n, "pair_n": n*len(qids),
                "qe_all_pairs_mean": float(part.mean()) if n else None,
                "qe_mean_over_queries_per_evidence": summary(part.mean(0).cpu().numpy()) if n else summary([]),
                "qe_max_over_queries_per_evidence": summary(part.max(0).values.cpu().numpy()) if n else summary([]),
                "exact_top20_slots": selected_n, "exact_top20_slot_share": selected_n/(20*len(qids)),
                "exact_top20_opportunity_rate": selected_n/(n*len(qids)) if n else None}
        sorted_scores = scores.sort(dim=1).values.contiguous()
        baseline_ranks = 1 + len(eids) - torch.searchsorted(sorted_scores, cached_sample_qe.contiguous(), right=True)
        qe_matrices[generator] = {"cached": cached_sample_qe.cpu(), "cached_rank": baseline_ranks.cpu()}
        for arm in ARMS:
            dest = cf[arm] if model is None else model.index_vector(cf[arm], "text")
            qe = qvec @ dest.T
            ranks = 1 + len(eids) - torch.searchsorted(sorted_scores, qe.contiguous(), right=True) - (cached_sample_qe > qe).long()
            qe_matrices[generator][arm] = qe.cpu()
            qe_matrices[generator][arm+"_rank"] = ranks.cpu()
        del scores, sorted_scores, part
        report = {}
        for scope in ("random_bin", "all_selected"):
            report[scope] = {}
            for b in BINS:
                ix = [i for i, r in enumerate(sample) if r["bucket"] == b and (scope == "all_selected" or scope in r["selection"])]
                old = qe_matrices[generator]["original"][:, ix].numpy()
                old_rank = qe_matrices[generator]["original_rank"][:, ix].numpy()
                originally_top = qe_matrices[generator]["cached_rank"][:, ix].numpy() <= 20
                report[scope][b] = {}
                for arm in ARMS:
                    new = qe_matrices[generator][arm][:, ix].numpy()
                    rank = qe_matrices[generator][arm+"_rank"][:, ix].numpy()
                    report[scope][b][arm] = {"sample_evidence_n": len(ix), "pair_n": int(new.size),
                        "qe": summary(new.ravel()), "qe_delta_vs_original": summary((new-old).ravel()),
                        "rank": summary(rank.ravel()), "rank_delta_vs_original": summary((rank-old_rank).ravel()),
                        "original_cached_top20_pairs": int(originally_top.sum()),
                        "on_original_top20_qe_delta": summary((new-old)[originally_top]),
                        "on_original_top20_new_rank": summary(rank[originally_top]),
                        "on_original_top20_survival": float((rank[originally_top] <= 20).mean()) if originally_top.any() else None}
        qe_reports[generator] = report
        print(json.dumps({"event": "exact_corpus_scoring", "generator": generator, "pairs": len(eids)*len(qids)}), flush=True)
    write_json(output / "all_corpus_scores.json", baseline_all)
    write_json(output / "counterfactual_qe_summary.json", qe_reports)
    torch.save({"queries": qids, "evidence": sids, "matrices": qe_matrices}, output / "counterfactual_qe_matrices.pt")
    # Same Q/E/T triples, with one E replaced at a time. Pool ranks are path
    # occurrence ranks among archived TEXT paths, never target retrieval ranks.
    path_reports = {}
    for generator, model in (("Qwen-Raw", None), ("B13", student)):
        all_paths = list(rows(output / f"paths_{generator}.jsonl.gz"))
        path_competitors = defaultdict(list)
        for p in all_paths:
            path_competitors[p["query_id"]].append(p["path"])
        path_competitors = {q: np.sort(v) for q, v in path_competitors.items()}
        paths = [p for p in all_paths if p["evidence_id"] in spos]
        del all_paths
        tids = sorted({p["target_id"] for p in paths})
        tpos = {t: i for i, t in enumerate(tids)}
        tt = torch.stack([store.embedding_features(t).embedding for t in tids]).to(device)
        tv = tt if model is None else model.index_vector(tt, "table")
        et_matrices = {arm: ((cf[arm] if model is None else model.relation_query(cf[arm], "text", "table")) @ tv.T).cpu().numpy() for arm in ARMS}
        qes = {arm: qe_matrices[generator][arm].numpy() for arm in ARMS}
        out_rows = []
        group_values = defaultdict(lambda: defaultdict(list))
        for p in paths:
            si, qi, ti = spos[p["evidence_id"]], qpos[p["query_id"]], tpos[p["target_id"]]
            comp = path_competitors[p["query_id"]]
            orig = float(qes["original"][qi, si] + et_matrices["original"][si, ti])
            original_rank = int(1+len(comp)-np.searchsorted(comp, orig, side="right")-(p["path"] > orig))
            interventions = {}
            for arm in ARMS:
                qe, et = float(qes[arm][qi, si]), float(et_matrices[arm][si, ti])
                score = qe+et
                rank = int(1+len(comp)-np.searchsorted(comp, score, side="right")-(p["path"] > score))
                interventions[arm] = {"qe": qe, "et": et, "path": score, "path_pool_rank": rank}
                for scope in ("all_sampled_paths", "strict_retained" if p["strict_target"] and p["retained"] else "other"):
                    v = group_values[(scope, p["bucket"], arm)]
                    v["qe"].append(qe)
                    v["et"].append(et)
                    v["et_delta"].append(et-float(et_matrices["original"][si, ti]))
                    v["path_delta"].append(score-orig)
                    v["path_rank_delta"].append(rank-original_rank)
            out_rows.append({**p, "interventions": interventions})
        write_rows(output / f"counterfactual_paths_{generator}.jsonl.gz", out_rows)
        path_reports[generator] = [{"scope": scope, "bucket": b, "arm": arm,
                                   **{name: summary(v) for name, v in values.items()}}
                                  for (scope, b, arm), values in group_values.items()]
        print(json.dumps({"event": "path_counterfactual", "generator": generator, "paths": len(paths)}), flush=True)
    write_json(output / "counterfactual_path_summary.json", path_reports)
    write_json(output / "ANALYSIS_COMPLETE.json", {"status": "complete", "queries": len(qids), "texts": len(eids),
        "sample": len(sample), "cache_parity": summary(parity), "exact_rank_ties": "competition rank (strictly greater scores), excluding replaced self",
        "code": fingerprint(Path(__file__)), "model_checkpoint": fingerprint(ps["student"]),
        "limitations": ["E-only counterfactual uses frozen Q/T and frozen competitors; not a full-corpus prompt migration",
            "Counterfactual path rank is within frozen text path occurrences, not target ranking or end-to-end Recall",
            "Strict-EO witness denotes a retained path to qrel target, not verified factual evidence",
            "Normal short content can be meaningful; length and prompt similarity alone do not establish semantic irrelevance",
            "Nonrandom hub/strict/placeholder enrichments are reported separately from random-bin samples"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("prepare", "encode", "analyze"), required=True)
    p.add_argument("--seed", type=int, default=260915)
    p.add_argument("--per-bin", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.phase == "prepare":
        prepare(args.root, args.output, args.seed, args.per_bin)
    elif args.phase == "encode":
        encode(args.root, args.output, args.device, args.batch_size)
    else:
        analyze(args.root, args.output, args.device)
