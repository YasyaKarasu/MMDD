"""MMDD Stage1 CLEAN-R1: fresh Teacher/Student on the raw EntiTables-20K lake.

Sole contract: ``mmdd_s1_clean_r1/EXPERIMENT_SPEC.zh-CN.md`` + ``clean_r1.json``.
No historical checkpoint, optimizer, PCA, teacher logit, candidate list or
ranking is read by any code path in this package.
"""

__all__ = ["reference", "config", "data", "cache", "models", "sampling", "retrieve", "train", "evaluate"]
