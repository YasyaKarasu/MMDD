"""Frozen Qwen3.5-9B loading shared by the selector reader and the recovery generator."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_qwen(model_dir: Path, *, seed: int, cpu_threads: int, processor_max_pixels: int | None = None
              ) -> tuple[Any, Any]:
    """bf16 + SDPA on ``cuda:0``, eval mode, no gradients. Returns ``(processor, model)``."""
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    extra = {} if processor_max_pixels is None else {"max_pixels": processor_max_pixels}
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, **extra)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    return processor, model
