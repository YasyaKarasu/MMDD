"""Consistent terminal progress bars for long-running research workflows."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a terminal-only progress bar with stable project defaults."""

    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("disable", None)
    return tqdm(iterable, **kwargs)
