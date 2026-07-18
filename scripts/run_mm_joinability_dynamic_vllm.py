#!/usr/bin/env python
"""Run multimodal joinability construction with dynamic vLLM GPU reallocation.

The runner starts one text vLLM server and one image/VL vLLM server. The builder
precomputes both model caches concurrently and writes one done marker per
modality. Whichever modality finishes first releases its GPU; this runner then
starts a second server for the remaining modality on that freed GPU. The builder
re-reads per-modality endpoint files before model requests, so the remaining
queue can use the new server without restarting.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import model_marker_protocol as model_markers
from build_wdc200k_mm_joinability_dataset import DiskGuard
from wdc200k_io import GuardedWriteTracker, PreWriteGuard
from wdc200k_runtime import (
    DEFAULT_MIN_FREE_DISK_BYTES,
    required_runtime_dir,
)

try:
    import requests
except ImportError:  # pragma: no cover - integration environment issue.
    requests = None  # type: ignore[assignment]


MODEL_MARKER_SCHEMA_VERSION = model_markers.MODEL_MARKER_SCHEMA_VERSION
MODEL_START_STAGE = model_markers.MODEL_START_STAGE
MODEL_READY_STAGE = model_markers.MODEL_READY_STAGE
MODEL_DONE_STAGE = model_markers.MODEL_DONE_STAGE


@dataclass(frozen=True)
class VllmServerSpec:
    role: str
    model_path: str
    served_model_name: str
    gpu: str
    port: int
    extra_args: list[str]
    host: str = "127.0.0.1"
    vllm_bin: str = "vllm"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> list[str]:
        return [
            self.vllm_bin,
            "serve",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.served_model_name,
            *self.extra_args,
        ]


def default_vllm_extra_args(args: argparse.Namespace) -> list[str]:
    if getattr(args, "no_default_vllm_memory_args", False):
        return []
    return [
        "--trust-remote-code",
        "--dtype",
        clean_arg_value(args.vllm_dtype),
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--gpu-memory-utilization",
        clean_arg_value(args.vllm_gpu_memory_utilization),
        "--enforce-eager",
        "--skip-mm-profiling",
        "--mm-processor-cache-gb",
        clean_arg_value(args.vllm_mm_processor_cache_gb),
        "--max-num-batched-tokens",
        str(args.vllm_max_num_batched_tokens),
        "--max-num-seqs",
        str(args.vllm_max_num_seqs),
    ]


def clean_arg_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_builder_command(
    *,
    python_executable: str,
    builder_script: Path,
    input_dir: Path,
    output_dir: Path,
    runtime_dir: Path,
    text_server: VllmServerSpec,
    primary_image_server: VllmServerSpec,
    text_endpoints_file: Path,
    image_endpoints_file: Path,
    model_start_marker: Path,
    model_ready_marker: Path,
    text_done_marker: Path,
    image_done_marker: Path,
    passthrough_args: list[str],
) -> list[str]:
    return [
        python_executable,
        str(builder_script),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--runtime_dir",
        str(runtime_dir),
        "--text_model_base_url",
        text_server.base_url,
        "--text_model_base_urls_file",
        str(text_endpoints_file),
        "--text_model_name",
        text_server.served_model_name,
        "--image_model_base_url",
        primary_image_server.base_url,
        "--image_model_base_urls_file",
        str(image_endpoints_file),
        "--image_model_name",
        primary_image_server.served_model_name,
        "--precompute_model_cache",
        "--model_start_marker",
        str(model_start_marker),
        "--model_ready_marker",
        str(model_ready_marker),
        "--model_text_done_marker",
        str(text_done_marker),
        "--model_image_done_marker",
        str(image_done_marker),
        *passthrough_args,
    ]


def process_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def start_server(spec: VllmServerSpec) -> subprocess.Popen[str]:
    return subprocess.Popen(
        spec.command(),
        env=process_env(spec.gpu),
        text=True,
        start_new_session=True,
        stdout=None,
        stderr=None,
    )


def stop_process(proc: subprocess.Popen[str] | None, *, timeout_seconds: float = 30.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=timeout_seconds)


def forward_signal_to_live_process_groups(
    signum: int,
    processes: Iterable[subprocess.Popen[str] | None],
) -> None:
    for process in processes:
        if process is None or process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            continue


class ForwardedSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def install_process_group_signal_handlers(
    processes: Callable[[], Iterable[subprocess.Popen[str] | None]],
) -> dict[int, object]:
    previous_handlers: dict[int, object] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        forward_signal_to_live_process_groups(signum, processes())
        raise ForwardedSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, handle_signal)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[int, object]) -> None:
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        previous = previous_handlers.get(signum)
        if previous is not None:
            signal.signal(signum, previous)


def mask_process_group_signals_for_cleanup() -> None:
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, signal.SIG_IGN)


def stop_processes_best_effort(
    processes: Iterable[subprocess.Popen[str] | None],
) -> list[tuple[subprocess.Popen[str], Exception]]:
    errors: list[tuple[subprocess.Popen[str], Exception]] = []
    for process in processes:
        if process is None:
            continue
        try:
            stop_process(process)
        except Exception as exc:
            errors.append((process, exc))
    return errors


def wait_for_server(base_url: str, *, timeout_seconds: float, poll_seconds: float = 2.0) -> None:
    if requests is None:
        raise RuntimeError("requests is required for vLLM health checks")
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/models", timeout=10.0)
            if int(getattr(response, "status_code", 500)) < 500:
                return
        except Exception as exc:  # pragma: no cover - integration only.
            last_error = exc
        time.sleep(poll_seconds)
    raise RuntimeError(f"Timed out waiting for {base_url}/models: {last_error}")


def write_endpoint_file(
    path: Path,
    urls: Iterable[str],
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        value = url.strip().rstrip("/")
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    encoded = "\n".join(deduped) + "\n"
    tracker = GuardedWriteTracker(path, pre_write_guard)
    tracker.before_write(len(encoded.encode("utf-8")))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        tracker.before_commit(0)
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_ready_marker(
    path: Path,
    *,
    context: model_markers.ModelMarkerContext | None = None,
    run_fingerprint: str | None = None,
    text_jobset_fingerprint: str | None = None,
    image_jobset_fingerprint: str | None = None,
    text_task_count: int | None = None,
    image_task_count: int | None = None,
    start_fingerprint: str | None = None,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    if context is None:
        identity = (
            run_fingerprint,
            text_jobset_fingerprint,
            image_jobset_fingerprint,
            text_task_count,
            image_task_count,
            start_fingerprint,
        )
        if any(value is None for value in identity):
            raise ValueError("ready marker identity is incomplete")
        context = model_markers.ModelMarkerContext(
            run_fingerprint=str(run_fingerprint),
            text_jobset_fingerprint=str(text_jobset_fingerprint),
            image_jobset_fingerprint=str(image_jobset_fingerprint),
            text_task_count=int(text_task_count),
            image_task_count=int(image_task_count),
            upstream_identities=(),
            start_fingerprint=str(start_fingerprint),
        )
    model_markers.atomic_write_json(
        Path(path),
        model_markers.ready_marker_payload(
            context,
            timestamp=time.time(),
        ),
        pre_write_guard=pre_write_guard,
    )


def read_pending_model_task_count(path: Path) -> int | None:
    context = model_markers.context_from_start_marker(path)
    if context is None:
        return None
    return context.text_task_count + context.image_task_count


def marker_matches_run(
    path: Path,
    expected_run_fingerprint: str | None,
    expected_jobset_fingerprint: str | None = None,
    *,
    expected_status: str | None = None,
    expected_model_kind: str | None = None,
    expected_task_count: int | None = None,
    expected_stage: str | None = None,
    expected_text_jobset_fingerprint: str | None = None,
    expected_image_jobset_fingerprint: str | None = None,
    expected_text_task_count: int | None = None,
    expected_image_task_count: int | None = None,
    expected_start_fingerprint: str | None = None,
) -> bool:
    inferred_stage = expected_stage
    if inferred_stage is None:
        inferred_stage = {
            "model_cache_ready_to_start": MODEL_START_STAGE,
            "vllm_servers_ready": MODEL_READY_STAGE,
            "text_model_cache_precomputed": MODEL_DONE_STAGE,
            "image_model_cache_precomputed": MODEL_DONE_STAGE,
        }.get(expected_status or "")
    if inferred_stage is None or expected_status is None:
        return False
    return model_markers.marker_matches(
        path,
        expected_stage=inferred_stage,
        expected_status=expected_status,
        model_kind=expected_model_kind,
        task_count=expected_task_count,
        run_fingerprint=expected_run_fingerprint,
        jobset_fingerprint=expected_jobset_fingerprint,
        text_jobset_fingerprint=expected_text_jobset_fingerprint,
        image_jobset_fingerprint=expected_image_jobset_fingerprint,
        text_task_count=expected_text_task_count,
        image_task_count=expected_image_task_count,
        start_fingerprint=expected_start_fingerprint,
    )


def read_marker_payload(path: Path) -> dict[str, object]:
    return model_markers.read_marker(path)


def wait_for_marker_or_builder_exit(
    *,
    marker: Path,
    builder: subprocess.Popen[str],
    timeout_seconds: float | None,
    poll_seconds: float = 2.0,
    expected_run_fingerprint: str | None = None,
    expected_status: str = "model_cache_ready_to_start",
) -> None:
    started = time.time()
    while not marker_matches_run(
        marker,
        expected_run_fingerprint,
        expected_status=expected_status,
    ):
        code = builder.poll()
        if code is not None:
            raise RuntimeError(f"Builder exited with code {code} before marker was written: {marker}")
        if timeout_seconds is not None and time.time() - started > timeout_seconds:
            raise RuntimeError(f"Timed out waiting for marker: {marker}")
        time.sleep(poll_seconds)


def wait_for_any_marker_or_builder_exit(
    *,
    markers: dict[str, Path],
    builder: subprocess.Popen[str],
    timeout_seconds: float | None,
    poll_seconds: float = 2.0,
    expected_run_fingerprint: str | None = None,
    expected_jobset_fingerprints: dict[str, str] | None = None,
    expected_task_counts: dict[str, int] | None = None,
    expected_start_fingerprint: str | None = None,
) -> set[str]:
    started = time.time()
    while True:
        completed = {
            kind
            for kind, marker in markers.items()
            if marker_matches_run(
                marker,
                expected_run_fingerprint,
                (expected_jobset_fingerprints or {}).get(kind),
                expected_status=f"{kind}_model_cache_precomputed",
                expected_model_kind=kind,
                expected_task_count=(
                    expected_task_counts or {}
                ).get(kind),
                expected_text_jobset_fingerprint=(
                    expected_jobset_fingerprints or {}
                ).get("text"),
                expected_image_jobset_fingerprint=(
                    expected_jobset_fingerprints or {}
                ).get("image"),
                expected_text_task_count=(
                    expected_task_counts or {}
                ).get("text"),
                expected_image_task_count=(
                    expected_task_counts or {}
                ).get("image"),
                expected_start_fingerprint=expected_start_fingerprint,
            )
        }
        if completed:
            return completed
        code = builder.poll()
        if code is not None:
            raise RuntimeError(f"Builder exited with code {code} before any model done marker was written")
        if timeout_seconds is not None and time.time() - started > timeout_seconds:
            marker_list = ", ".join(str(path) for path in markers.values())
            raise RuntimeError(f"Timed out waiting for model done markers: {marker_list}")
        time.sleep(poll_seconds)


def passthrough_has_arg(passthrough_args: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in passthrough_args)


def passthrough_option_value(
    passthrough_args: list[str],
    option: str,
) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(passthrough_args):
        token = passthrough_args[index]
        if token == option:
            if index + 1 >= len(passthrough_args):
                raise ValueError(f"{option} requires a value")
            values.append(passthrough_args[index + 1])
            index += 2
            continue
        if token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ValueError(f"{option} may be provided at most once")
    return values[0] if values else None


_RESERVED_BUILDER_OPTIONS = (
    "--input_dir",
    "--output_dir",
    "--runtime_dir",
    "--text_model_base_url",
    "--text_model_base_urls_file",
    "--text_model_name",
    "--image_model_base_url",
    "--image_model_base_urls_file",
    "--image_model_name",
    "--run_fingerprint",
    "--model_start_marker",
    "--model_ready_marker",
    "--model_text_done_marker",
    "--model_image_done_marker",
)


def validate_builder_passthrough(passthrough_args: list[str]) -> None:
    for option in _RESERVED_BUILDER_OPTIONS:
        if passthrough_has_arg(passthrough_args, option):
            raise ValueError(
                f"{option} is controlled by the dynamic runner"
            )
    passthrough_option_value(passthrough_args, "--work_dir")
    minimum = passthrough_option_value(
        passthrough_args,
        "--min_free_disk_bytes",
    )
    if minimum is not None and int(minimum) < 0:
        raise ValueError("--min_free_disk_bytes must be non-negative")


def with_default_model_workers(passthrough_args: list[str], workers: int) -> list[str]:
    if workers <= 0:
        return passthrough_args
    result = list(passthrough_args)
    for option in ("--text_model_workers", "--image_model_workers"):
        if not passthrough_has_arg(result, option):
            result.extend([option, str(workers)])
    return result


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Start vLLM servers and run build_mm_joinability_dataset.py with modality-agnostic dynamic GPU reallocation.",
        allow_abbrev=False,
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_model_path", required=True)
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_gpu", default="1")
    parser.add_argument("--text_port", type=int, default=8001)
    parser.add_argument("--secondary_text_port", type=int, default=8003)
    parser.add_argument("--image_model_path", required=True)
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Thinking")
    parser.add_argument("--primary_image_gpu", default="0")
    parser.add_argument("--primary_image_port", type=int, default=8000)
    parser.add_argument("--secondary_image_gpu", default="1")
    parser.add_argument("--secondary_image_port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--vllm_bin", default="vllm")
    parser.add_argument("--server_start_timeout_seconds", type=float, default=900.0)
    parser.add_argument("--model_start_timeout_seconds", type=float, default=None, help="Maximum seconds to wait for the builder to finish Wikipedia/material preparation before vLLM startup. Default waits indefinitely.")
    parser.add_argument("--run_fingerprint", default="", help="Optional staged-run identity used to fence stale model markers.")
    parser.add_argument("--runtime_dir", default="", help="Marker/endpoint directory. Must equal the WDC builder work directory plus /runtime.")
    parser.add_argument("--first_done_timeout_seconds", type=float, default=None)
    parser.add_argument("--text_done_timeout_seconds", type=float, default=None, help="Deprecated alias for --first_done_timeout_seconds.")
    parser.add_argument("--dynamic_model_workers", type=int, default=2, help="Default per-modality builder workers unless overridden in passthrough args. Use 0 to leave builder defaults unchanged.")
    parser.add_argument("--builder_script", default=str(Path(__file__).with_name("build_wdc200k_mm_joinability_dataset.py")))
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--vllm_dtype", default="bfloat16")
    parser.add_argument("--vllm_max_model_len", type=int, default=8192)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=1024)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=1)
    parser.add_argument("--vllm_mm_processor_cache_gb", type=float, default=0)
    parser.add_argument("--no_default_vllm_memory_args", action="store_true", help="Do not apply the conservative vLLM memory defaults copied from the known-good manual sessions.")
    parser.add_argument("--vllm_extra_arg", action="append", default=[], help="Extra argument applied to all vLLM serve commands. Repeat for multiple tokens.")
    parser.add_argument("--text_vllm_extra_arg", action="append", default=[], help="Extra argument applied only to the text vLLM server.")
    parser.add_argument("--image_vllm_extra_arg", action="append", default=[], help="Extra argument applied only to both image vLLM servers.")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, passthrough_args = parse_args(argv)
    validate_builder_passthrough(passthrough_args)
    output_dir = Path(args.output_dir).resolve()
    work_dir_value = passthrough_option_value(
        passthrough_args,
        "--work_dir",
    )
    expected_runtime_dir = required_runtime_dir(output_dir, work_dir_value)
    runtime_dir = (
        Path(args.runtime_dir).resolve()
        if args.runtime_dir
        else expected_runtime_dir
    )
    if runtime_dir != expected_runtime_dir:
        raise ValueError(
            "runtime_dir must equal work_dir/runtime: "
            f"{runtime_dir} != {expected_runtime_dir}"
        )
    minimum_value = passthrough_option_value(
        passthrough_args,
        "--min_free_disk_bytes",
    )
    minimum_free_disk_bytes = (
        int(minimum_value)
        if minimum_value is not None
        else DEFAULT_MIN_FREE_DISK_BYTES
    )
    runtime_guard = DiskGuard(minimum_free_disk_bytes)
    text_endpoints_file = runtime_dir / "text_endpoints.txt"
    image_endpoints_file = runtime_dir / "image_endpoints.txt"
    model_start_marker = runtime_dir / "model_start.json"
    model_ready_marker = runtime_dir / "model_ready.json"
    text_done_marker = runtime_dir / "text_done.json"
    image_done_marker = runtime_dir / "image_done.json"
    first_done_timeout = args.first_done_timeout_seconds
    if first_done_timeout is None:
        first_done_timeout = args.text_done_timeout_seconds

    common_extra = [*default_vllm_extra_args(args), *args.vllm_extra_arg]
    text_server = VllmServerSpec(
        role="text",
        model_path=args.text_model_path,
        served_model_name=args.text_model_name,
        gpu=args.text_gpu,
        port=args.text_port,
        host=args.host,
        vllm_bin=args.vllm_bin,
        extra_args=[*common_extra, *args.text_vllm_extra_arg],
    )
    primary_image_server = VllmServerSpec(
        role="image-primary",
        model_path=args.image_model_path,
        served_model_name=args.image_model_name,
        gpu=args.primary_image_gpu,
        port=args.primary_image_port,
        host=args.host,
        vllm_bin=args.vllm_bin,
        extra_args=[*common_extra, *args.image_vllm_extra_arg],
    )
    secondary_image_server = VllmServerSpec(
        role="image-secondary",
        model_path=args.image_model_path,
        served_model_name=args.image_model_name,
        gpu=args.secondary_image_gpu,
        port=args.secondary_image_port,
        host=args.host,
        vllm_bin=args.vllm_bin,
        extra_args=[*common_extra, *args.image_vllm_extra_arg],
    )
    secondary_text_server = VllmServerSpec(
        role="text-secondary",
        model_path=args.text_model_path,
        served_model_name=args.text_model_name,
        gpu=args.primary_image_gpu,
        port=args.secondary_text_port,
        host=args.host,
        vllm_bin=args.vllm_bin,
        extra_args=[*common_extra, *args.text_vllm_extra_arg],
    )

    text_proc: subprocess.Popen[str] | None = None
    primary_image_proc: subprocess.Popen[str] | None = None
    secondary_text_proc: subprocess.Popen[str] | None = None
    secondary_image_proc: subprocess.Popen[str] | None = None
    builder_proc: subprocess.Popen[str] | None = None
    previous_signal_handlers = install_process_group_signal_handlers(
        lambda: (
            builder_proc,
            text_proc,
            primary_image_proc,
            secondary_text_proc,
            secondary_image_proc,
        )
    )
    try:
        runtime_guard(runtime_dir, 0)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for marker in (model_start_marker, model_ready_marker, text_done_marker, image_done_marker):
            if marker.exists():
                marker.unlink()
        write_endpoint_file(
            text_endpoints_file,
            [text_server.base_url],
            pre_write_guard=runtime_guard,
        )
        write_endpoint_file(
            image_endpoints_file,
            [primary_image_server.base_url],
            pre_write_guard=runtime_guard,
        )

        builder_passthrough_args = with_default_model_workers(passthrough_args, args.dynamic_model_workers)
        if args.run_fingerprint and not passthrough_has_arg(
            builder_passthrough_args,
            "--run_fingerprint",
        ):
            builder_passthrough_args.extend(
                ["--run_fingerprint", args.run_fingerprint]
            )
        builder_command = build_builder_command(
            python_executable=args.python_executable,
            builder_script=Path(args.builder_script),
            input_dir=Path(args.input_dir),
            output_dir=output_dir,
            runtime_dir=runtime_dir,
            text_server=text_server,
            primary_image_server=primary_image_server,
            text_endpoints_file=text_endpoints_file,
            image_endpoints_file=image_endpoints_file,
            model_start_marker=model_start_marker,
            model_ready_marker=model_ready_marker,
            text_done_marker=text_done_marker,
            image_done_marker=image_done_marker,
            passthrough_args=builder_passthrough_args,
        )
        builder_proc = subprocess.Popen(builder_command, text=True, start_new_session=True)
        wait_for_marker_or_builder_exit(
            marker=model_start_marker,
            builder=builder_proc,
            timeout_seconds=args.model_start_timeout_seconds,
            expected_run_fingerprint=args.run_fingerprint,
        )
        marker_context = model_markers.context_from_start_marker(
            model_start_marker
        )
        if marker_context is None:
            raise RuntimeError(
                f"Model start marker became invalid: {model_start_marker}"
            )
        expected_jobsets = {
            "text": marker_context.text_jobset_fingerprint,
            "image": marker_context.image_jobset_fingerprint,
        }
        expected_task_counts = {
            "text": marker_context.text_task_count,
            "image": marker_context.image_task_count,
        }
        start_fingerprint = marker_context.start_fingerprint

        if read_pending_model_task_count(model_start_marker) == 0:
            return int(builder_proc.wait())

        text_proc = start_server(text_server)
        primary_image_proc = start_server(primary_image_server)
        wait_for_server(text_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
        wait_for_server(primary_image_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
        write_ready_marker(
            model_ready_marker,
            context=marker_context,
            pre_write_guard=runtime_guard,
        )

        completed = wait_for_any_marker_or_builder_exit(
            markers={"text": text_done_marker, "image": image_done_marker},
            builder=builder_proc,
            timeout_seconds=first_done_timeout,
            expected_run_fingerprint=args.run_fingerprint,
            expected_jobset_fingerprints=expected_jobsets,
            expected_task_counts=expected_task_counts,
            expected_start_fingerprint=start_fingerprint,
        )

        if completed == {"text"} and not marker_matches_run(
            image_done_marker,
            args.run_fingerprint,
            expected_jobsets.get("image"),
            expected_status="image_model_cache_precomputed",
            expected_model_kind="image",
            expected_task_count=expected_task_counts["image"],
            expected_text_jobset_fingerprint=expected_jobsets["text"],
            expected_image_jobset_fingerprint=expected_jobsets["image"],
            expected_text_task_count=expected_task_counts["text"],
            expected_image_task_count=expected_task_counts["image"],
            expected_start_fingerprint=start_fingerprint,
        ):
            stop_process(text_proc)
            text_proc = None
            secondary_image_proc = start_server(secondary_image_server)
            wait_for_server(secondary_image_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
            write_endpoint_file(
                image_endpoints_file,
                [
                    primary_image_server.base_url,
                    secondary_image_server.base_url,
                ],
                pre_write_guard=runtime_guard,
            )
        elif completed == {"image"} and not marker_matches_run(
            text_done_marker,
            args.run_fingerprint,
            expected_jobsets.get("text"),
            expected_status="text_model_cache_precomputed",
            expected_model_kind="text",
            expected_task_count=expected_task_counts["text"],
            expected_text_jobset_fingerprint=expected_jobsets["text"],
            expected_image_jobset_fingerprint=expected_jobsets["image"],
            expected_text_task_count=expected_task_counts["text"],
            expected_image_task_count=expected_task_counts["image"],
            expected_start_fingerprint=start_fingerprint,
        ):
            stop_process(primary_image_proc)
            primary_image_proc = None
            secondary_text_proc = start_server(secondary_text_server)
            wait_for_server(secondary_text_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
            write_endpoint_file(
                text_endpoints_file,
                [
                    text_server.base_url,
                    secondary_text_server.base_url,
                ],
                pre_write_guard=runtime_guard,
            )

        return int(builder_proc.wait())
    except ForwardedSignal as exc:
        return 128 + exc.signum
    finally:
        mask_process_group_signals_for_cleanup()
        try:
            cleanup_errors = stop_processes_best_effort(
                (
                    builder_proc,
                    text_proc,
                    primary_image_proc,
                    secondary_text_proc,
                    secondary_image_proc,
                )
            )
        finally:
            restore_signal_handlers(previous_signal_handlers)
        for process, error in cleanup_errors:
            print(
                f"warning: failed to stop process group {process.pid}: {error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
