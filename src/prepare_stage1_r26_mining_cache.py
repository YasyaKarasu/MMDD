"""Copy immutable T0 pair scores by query so priority mining has its own writer."""
import json
from pathlib import Path
import sqlite3
import time

from prepare_stage1_r26 import OUT, file_record
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json


def run() -> dict:
    directory = OUT / "feedback"
    parent = directory / "B13/MINING_RECEIPT.json"
    baseline = json.loads(parent.read_text())
    if baseline["queries"] != 11390 or baseline["positive_mislabeled_as_hard"]:
        raise ValueError("Finish valid full B13 mining before the priority-cache snapshot")
    source = directory / "train_T0_pairs.sqlite"
    target = directory / "train_T0_pairs_priority.sqlite"
    receipt = directory / "PRIORITY_CACHE_COPY.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    if target.exists():
        raise ValueError("An unaudited priority cache exists; inspect before reuse")
    started = time.monotonic()
    counts = {baseline["namespace"]:0}
    queries = [r["query_id"] for r in read_rows(OUT / "common/feedback_queries.jsonl")]
    with sqlite3.connect(f"file:{source}?mode=ro",uri=True) as reader, sqlite3.connect(target) as writer:
        writer.execute("CREATE TABLE scores (namespace TEXT,q TEXT,t TEXT,qsha TEXT,tsha TEXT,score REAL,PRIMARY KEY(namespace,q,t,qsha,tsha))")
        for index,q in enumerate(queries,1):
            # Each read finishes promptly; the original writer can continue.
            # Scores are independent immutable functions of the full pair key.
            rows = list(reader.execute("SELECT namespace,q,t,qsha,tsha,score FROM scores WHERE namespace=? AND q=?",(baseline["namespace"],q)))
            if not rows:
                raise ValueError("Completed B13 query lacks cached T0 scores")
            writer.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?)",rows)
            counts[baseline["namespace"]] += len(rows)
            if index % 128 == 0:
                writer.commit()
            if index % 1000 == 0:
                print(json.dumps({"copied_queries":index,"pairs":sum(counts.values())}),flush=True)
    result = {"execution_status":"ran","scientific_validity":"valid",
              "source_live_database":str(source),"priority_working_database":str(target),
              "initial_copy_sha256":file_record(target)["sha256"],"initial_namespace_pair_counts":counts,"queries":len(queries),
              "baseline":file_record(parent),"elapsed_seconds":time.monotonic()-started,
              "semantics":"Source opened read-only. Copy immutable scores separately per query, retaining exact Teacher/Q/T/actual-feature hashes. Concurrent source INSERTs may add reusable pairs but cannot alter their semantic identity. No global database snapshot is needed for a pure score cache. Priority mining alone writes the new database. Initial hash describes copy time, not the later mutable cache; CPU scoring unchanged.",
              "code":file_record(Path(__file__))}
    _json(receipt,result)
    return result


if __name__ == "__main__":
    print(json.dumps(run()))
