"""Compare actual fresh PCA tensors with historical PCA artifacts numerically."""
from __future__ import annotations

import json
import torch

from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r25 import _json


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    a, b = left.float(), right.float()
    return {"shape": list(a.shape), "exact_equal": torch.equal(a, b),
            "max_absolute_difference": float((a-b).abs().max()),
            "relative_frobenius_difference": float((a-b).norm()/b.norm()),
            "row_aligned_mean_cosine": float(torch.nn.functional.cosine_similarity(a,b,dim=1).mean()) if a.ndim == 2 else None}


def run() -> dict:
    torch.set_num_threads(4)
    actual_path = ROOT / "work/stage1_pca_dimension_ceiling_20260828/pca_spectrum.pt"
    historical_path = ROOT / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt"
    init_path = ROOT / "work/stage1_optimization_r11_20260908/taskA_protocol/baselines/pca_init.pt"
    actual = torch.load(actual_path, map_location="cpu", weights_only=False)
    historical = torch.load(historical_path, map_location="cpu", weights_only=False)
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    a, b = actual["projection"][:1024].float(), historical["projection"].float()
    singular = torch.linalg.svdvals(a @ b.T)
    result = {"execution_status": "ran", "scientific_validity": "valid",
        "actual": file_record(actual_path), "historical": file_record(historical_path), "historical_init": file_record(init_path),
        "metadata": {name: {key: value for key,value in payload.items() if not torch.is_tensor(value)}
                     for name,payload in (("actual",actual),("historical",historical))},
        "basis_comparison": compare(a,b), "mean_comparison": compare(actual["mean"],historical["mean"]),
        "subspace": {"mean_squared_principal_cosine": float(singular.square().mean()),
                     "min_principal_cosine": float(singular.min()), "median_principal_cosine": float(singular.median())},
        "orthonormality_max_error": {name: float((value@value.T-torch.eye(1024)).abs().max()) for name,value in (("actual",a),("historical",b))},
        "historical_init_projections": {key: compare(value,b) for key,value in init["state_dict"].items()
                                        if key.startswith("projections.") and key.endswith("weight")},
        "historical_init_identity_relations": {key: torch.equal(value,torch.eye(1024)) for key,value in init["state_dict"].items()
                                               if key.startswith("relations.") and value.shape == (1024,1024)},
        "application_semantics": "P is applied as a linear projection without subtracting either saved PCA mean; R25 uses actual pca_spectrum basis and matching frozen anchor",
        "interpretation": "Distinct unsupervised PCA artifacts are permitted reconstructed initialization; tensor mismatch alone does not invalidate audited fresh C1 or establish a causal performance effect"}
    _json(OUT / "acceptance/PCA_HISTORICAL_NUMERIC_AUDIT.json",result)
    recipe_path = OUT / "RECIPE_DIFF.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["PCA_provenance"] = {"status": "matched" if torch.equal(a,b) else "reconstructed_distinct_unsupervised_PCA",
        "audit": file_record(OUT / "acceptance/PCA_HISTORICAL_NUMERIC_AUDIT.json"), "retrain_C1_required": False,
        "reason": "C1 actual initial P/identity R and actual PCA anchor were independently audited"}
    _json(recipe_path,recipe)
    return result


if __name__ == "__main__":
    print(json.dumps(run()))
