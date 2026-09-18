"""Wall-clock and high-resolution timing instrumentation for CLEAN-R1.

The spec asks for per-object encoding cost, training cost per epoch, index build
cost and online retrieval latency, each broken out rather than rolled into one
command duration.  ``Timing`` accumulates named measurements with counts and a
percentile summary, and can borrow the clock from another accumulator so nested
stages (preprocess inside encode inside the whole cache build) are comparable.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any, Iterator


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


class Timing:
    """Accumulates named duration samples into a reportable summary."""

    def __init__(self, clock: "Timing | None" = None) -> None:
        self.samples: dict[str, list[float]] = {}
        # When this Timing is a sub-clock of another, exposing the parent lets a
        # caller subtract the attributed share from a total (the remainder is
        # un-attributed overhead, not missing time).
        self.parent = clock
        if clock is not None:
            clock.children.append(self)
        self.children: list["Timing"] = []

    def record(self, stage: str, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"negative duration for {stage}: {seconds}")
        self.samples.setdefault(stage, []).append(float(seconds))

    @contextmanager
    def stage(self, name: str, **labels: Any) -> Iterator[None]:
        """Time one block.  Labels become a ``name:key=value`` suffix."""
        key = name
        if labels:
            key = name + ":" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(key, time.perf_counter() - start)

    @contextmanager
    def clock(self, name: str, **labels: Any) -> Iterator["Timing"]:
        """A child Timing whose total is also recorded under ``name`` on self."""
        child = Timing(self)
        start = time.perf_counter()
        try:
            yield child
        finally:
            self.record(name, time.perf_counter() - start)

    def child(self) -> "Timing":
        return Timing(self)

    def merge(self, other: "Timing", prefix: str = "") -> None:
        for key, values in other.samples.items():
            self.samples.setdefault(prefix + key, []).extend(values)

    def total(self, stage: str) -> float:
        return float(sum(self.samples.get(stage, [])))

    def count(self, stage: str) -> int:
        return len(self.samples.get(stage, []))

    def attributed_total(self) -> float:
        """Sum over every stage recorded directly on this accumulator."""
        return float(sum(sum(values) for values in self.samples.values()))

    def summarise(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, values in sorted(self.samples.items()):
            ordered = sorted(values)
            out[key] = {
                "count": len(ordered),
                "total_seconds": float(sum(ordered)),
                "mean_seconds": float(sum(ordered) / len(ordered)),
                "p50_seconds": _percentile(ordered, 0.50),
                "p95_seconds": _percentile(ordered, 0.95),
                "max_seconds": ordered[-1],
            }
        return out

    def report(self, *, name: str, total_seconds: float | None = None) -> dict[str, Any]:
        """A serialisable report; children are nested under ``children``."""
        payload: dict[str, Any] = {"name": name, "stages": self.summarise()}
        if total_seconds is not None:
            attributed = self.attributed_total()
            payload["total_seconds"] = float(total_seconds)
            payload["attributed_seconds"] = attributed
            payload["unattributed_seconds"] = float(total_seconds) - attributed
        if self.children:
            payload["children"] = [
                child.report(name=f"child_{index}") for index, child in enumerate(self.children)
            ]
        return payload


def timed(timing: Timing, stage: str, **labels: Any):
    """Decorator form of :meth:`Timing.stage` for a whole function."""

    def decorator(function):
        def wrapper(*args, **kwargs):
            with timing.stage(stage, **labels):
                return function(*args, **kwargs)

        wrapper.__name__ = getattr(function, "__name__", "wrapped")
        wrapper.__doc__ = function.__doc__
        return wrapper

    return decorator
