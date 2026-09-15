"""Hash actual frozen feature files/backbones and retain small schema samples."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil

from package_stage1_r26 import digest, external_record, safe_path
from run_stage1_r25 import _r25_teacher_feature_paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_optimization_r26_20260914"


def run() -> dict:
    destination = OUT / "external_inputs"
    destination.mkdir(exist_ok=True)
    roots = list(dict.fromkeys([ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
                               *_r25_teacher_feature_paths(ROOT)]))
    manifests, files, examples, missing = [], {}, {}, []
    for priority,root in enumerate(roots):
        for name in ("manifest.jsonl","teacher_manifest.jsonl"):
            path = root / name
            if not path.exists():
                continue
            saved = destination / "manifests" / f"root{priority:02d}" / name
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,saved)
            manifests.append({"priority":priority,"root":str(root),"path":str(path),"bytes":path.stat().st_size,
                              "sha256":digest(path),"included_copy":str(saved)})
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    for key in ("feature_path","teacher_feature_path"):
                        if not row.get(key):
                            continue
                        source = safe_path(root / row[key])
                        if not source.is_file():
                            missing.append({"manifest":str(path),"object_id":row["object_id"],"path":str(source)})
                            continue
                        files.setdefault(source,{"object_id":row["object_id"],"object_type":row["object_type"],"field":key,
                                                 "manifest":str(path),"source_fingerprint":row.get("source_fingerprint")})
                        examples.setdefault((priority,key,row["object_type"]),source)
    total_bytes = 0
    hashes = destination / "feature_files.jsonl.gz"
    with gzip.open(hashes,"wt") as stream:
        for i,(path,metadata) in enumerate(files.items(),1):
            before = path.stat()
            checksum = digest(path)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise ValueError(f"Frozen feature changed while hashing: {path}")
            stream.write(json.dumps({**metadata,"path":str(path),"bytes":before.st_size,"sha256":checksum})+"\n")
            total_bytes += before.st_size
            if i % 10000 == 0:
                print(json.dumps({"hashed_feature_files":i,"total_files":len(files),"bytes":total_bytes}),flush=True)
    samples = []
    for (priority,field,kind),path in examples.items():
        record = external_record(path)
        record.update({"root_priority":priority,"field":field,"object_type":kind})
        if path.stat().st_size <= 8*1024**2:
            sample = destination / "tensor_samples" / f"root{priority:02d}_{field}_{kind}.pt"
            sample.parent.mkdir(exist_ok=True)
            shutil.copyfile(path,sample)
            record["included_sample"] = str(sample)
        samples.append(record)
    backbones = []
    for model in ("Qwen3.5-9B","Qwen3-VL-Embedding-8B"):
        directory = ROOT / "hf_models" / model
        for path in sorted(directory.iterdir()):
            if path.name == ".env.openai" or path.suffix not in (".safetensors",".json",".txt",".model",".jinja"):
                continue
            path = safe_path(path)
            record = {"model":model,"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)}
            if path.suffix == ".safetensors":
                with path.open("rb") as stream:
                    length = int.from_bytes(stream.read(8),"little")
                    record["schema"] = json.loads(stream.read(length))
            else:
                saved = destination / "model_configuration" / model / path.name
                saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,saved)
                record["included_copy"] = str(saved)
            backbones.append(record)
            print(json.dumps({"hashed_model_file":model+"/"+path.name}),flush=True)
    result = {"execution_status":"ran","scientific_validity":"valid" if not missing else "requires_missing_entry_review",
              "feature_files":len(files),"feature_bytes":total_bytes,"missing_manifest_entries":missing,
              "feature_file_inventory":{"path":str(hashes),"bytes":hashes.stat().st_size,"sha256":digest(hashes)},
              "feature_manifests":manifests,"schema_samples":samples,"backbones":backbones,
              "scope":"Every existing file referenced by the actual frozen feature/Teacher manifests is individually hashed; missing entries explicit. Schema examples cover each available root/field/modality; sequence shapes vary by object. Backbone safetensors include complete headers. Small actual tensor/config samples are included; large external bytes are referenced.",
              "code":{"path":str(Path(__file__)),"sha256":digest(Path(__file__))}}
    (destination / "EXTERNAL_INPUTS_RECEIPT.json").write_text(json.dumps(result,indent=2)+"\n")
    return {"feature_files":len(files),"feature_GiB":total_bytes/1024**3,"missing_entries":len(missing),"backbone_files":len(backbones)}


if __name__ == "__main__":
    print(json.dumps(run()))
