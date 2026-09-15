"""Full-union column promotion with a query-balanced natural train CDF."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ColumnSimilarity:
    def __init__(self, vectors: dict[str,torch.Tensor], device: torch.device):
        groups = {}
        for key,value in vectors.items():
            role,table,_ = key.split(":",2)
            groups.setdefault((role,table),[]).append(value)
        self.groups = {key:F.normalize(torch.stack(values).float(),dim=1) for key,values in groups.items()}
        self.device = device
        self.cache: dict[str,dict[str,float | None]] = {}

    @torch.inference_mode()
    def scores(self, query: str, targets: list[str]) -> dict[str,float | None]:
        scores = self.cache.setdefault(query,{})
        missing = [t for t in dict.fromkeys(targets) if t not in scores]
        q = self.groups.get(("query",query))
        valid = []
        for t in missing:
            if q is None or ("target",t) not in self.groups:
                scores[t] = None
            else:
                valid.append(t)
        for start in range(0,len(valid),256):
            batch = valid[start:start+256]
            matrix = torch.cat([self.groups["target",t] for t in batch]).to(self.device)
            maxima = (q.to(self.device) @ matrix.T).amax(0).cpu()
            offset = 0
            for t in batch:
                count = len(self.groups["target",t])
                scores[t] = float(maxima[offset:offset+count].max())
                offset += count
        return {t:scores[t] for t in targets}


class QueryBalancedCDF:
    def __init__(self, per_query: list[list[float]]):
        if not per_query or any(not values for values in per_query):
            raise ValueError("Every frozen reference query needs observed natural-pool similarities")
        values = np.array([v for row in per_query for v in row])
        weights = np.array([1/(len(per_query)*len(row)) for row in per_query for _ in row])
        order = np.argsort(values,kind="stable")
        self.values = values[order]
        self.cumulative = np.cumsum(weights[order])

    def alpha(self, score: float | None) -> float:
        if score is None:
            return 1.
        position = np.searchsorted(self.values,score,side="right")
        return 1-float(np.clip(self.cumulative[position-1],0,1)) if position else 1.
