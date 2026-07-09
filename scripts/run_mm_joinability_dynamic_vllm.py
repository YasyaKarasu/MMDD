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
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover - integration environment issue.
    requests = None  # type: ignore[assignment]


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
    text_server: VllmServerSpec,
    primary_image_server: VllmServerSpec,
    text_endpoints_file: Path,
    image_endpoints_file: Path,
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


def write_endpoint_file(path: Path, urls: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        value = url.strip().rstrip("/")
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    path.write_text("\n".join(deduped) + "\n", encoding="utf-8")


def wait_for_marker_or_builder_exit(
    *,
    marker: Path,
    builder: subprocess.Popen[str],
    timeout_seconds: float | None,
    poll_seconds: float = 2.0,
) -> None:
    started = time.time()
    while not marker.exists():
        code = builder.poll()
        if code is not None:
            raise RuntimeError(f"Builder exited with code {code} before text done marker was written")
        if timeout_seconds is not None and time.time() - started > timeout_seconds:
            raise RuntimeError(f"Timed out waiting for text done marker: {marker}")
        time.sleep(poll_seconds)


def wait_for_any_marker_or_builder_exit(
    *,
    markers: dict[str, Path],
    builder: subprocess.Popen[str],
    timeout_seconds: float | None,
    poll_seconds: float = 2.0,
) -> set[str]:
    started = time.time()
    while True:
        completed = {kind for kind, marker in markers.items() if marker.exists()}
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
    parser.add_argument("--first_done_timeout_seconds", type=float, default=None)
    parser.add_argument("--text_done_timeout_seconds", type=float, default=None, help="Deprecated alias for --first_done_timeout_seconds.")
    parser.add_argument("--dynamic_model_workers", type=int, default=2, help="Default per-modality builder workers unless overridden in passthrough args. Use 0 to leave builder defaults unchanged.")
    parser.add_argument("--builder_script", default=str(Path(__file__).with_name("build_mm_joinability_dataset.py")))
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
    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "_dynamic_vllm"
    text_endpoints_file = runtime_dir / "text_endpoints.txt"
    image_endpoints_file = runtime_dir / "image_endpoints.txt"
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
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for marker in (text_done_marker, image_done_marker):
            if marker.exists():
                marker.unlink()
        write_endpoint_file(text_endpoints_file, [text_server.base_url])
        write_endpoint_file(image_endpoints_file, [primary_image_server.base_url])

        text_proc = start_server(text_server)
        primary_image_proc = start_server(primary_image_server)
        wait_for_server(text_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
        wait_for_server(primary_image_server.base_url, timeout_seconds=args.server_start_timeout_seconds)

        builder_passthrough_args = with_default_model_workers(passthrough_args, args.dynamic_model_workers)
        builder_command = build_builder_command(
            python_executable=args.python_executable,
            builder_script=Path(args.builder_script),
            input_dir=Path(args.input_dir),
            output_dir=output_dir,
            text_server=text_server,
            primary_image_server=primary_image_server,
            text_endpoints_file=text_endpoints_file,
            image_endpoints_file=image_endpoints_file,
            text_done_marker=text_done_marker,
            image_done_marker=image_done_marker,
            passthrough_args=builder_passthrough_args,
        )
        builder_proc = subprocess.Popen(builder_command, text=True, start_new_session=True)
        completed = wait_for_any_marker_or_builder_exit(
            markers={"text": text_done_marker, "image": image_done_marker},
            builder=builder_proc,
            timeout_seconds=first_done_timeout,
        )

        if completed == {"text"} and not image_done_marker.exists():
            stop_process(text_proc)
            text_proc = None
            secondary_image_proc = start_server(secondary_image_server)
            wait_for_server(secondary_image_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
            write_endpoint_file(image_endpoints_file, [primary_image_server.base_url, secondary_image_server.base_url])
        elif completed == {"image"} and not text_done_marker.exists():
            stop_process(primary_image_proc)
            primary_image_proc = None
            secondary_text_proc = start_server(secondary_text_server)
            wait_for_server(secondary_text_server.base_url, timeout_seconds=args.server_start_timeout_seconds)
            write_endpoint_file(text_endpoints_file, [text_server.base_url, secondary_text_server.base_url])

        return int(builder_proc.wait())
    finally:
        if builder_proc is not None and builder_proc.poll() is None:
            stop_process(builder_proc)
        stop_process(text_proc)
        stop_process(primary_image_proc)
        stop_process(secondary_text_proc)
        stop_process(secondary_image_proc)


if __name__ == "__main__":
    raise SystemExit(main())
