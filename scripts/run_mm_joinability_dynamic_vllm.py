#!/usr/bin/env python
"""Run EntiTables construction with dynamic, priority-owned local vLLM.

In round mode the runner starts services only at a model-round handshake and
stops every service before the next network-only phase. An optional filesystem
protocol lets a preemptible WDC sidecar borrow both GPUs between those rounds.
Within a round, the first completed modality can still donate its GPU to a
second server for the remaining modality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from gpu_priority_protocol import PriorityGpuOwner

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


class ServerExitedBeforeHealthy(RuntimeError):
    """Raised when a vLLM child exits before its health endpoint is ready."""


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
    model_start_marker: Path,
    model_ready_marker: Path,
    text_done_marker: Path,
    image_done_marker: Path,
    model_round_control_dir: Path,
    model_round_run_id: str,
    passthrough_args: list[str],
    remote_layout_args: list[str] | None = None,
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
        "--model_start_marker",
        str(model_start_marker),
        "--model_ready_marker",
        str(model_ready_marker),
        "--model_text_done_marker",
        str(text_done_marker),
        "--model_image_done_marker",
        str(image_done_marker),
        "--model_round_control_dir",
        str(model_round_control_dir),
        "--model_round_run_id",
        model_round_run_id,
        *(remote_layout_args or []),
        *passthrough_args,
    ]


def process_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def start_server(spec: VllmServerSpec, *, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.Popen(
            spec.command(),
            env=process_env(spec.gpu),
            text=True,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def _cleanup_warning(proc: subprocess.Popen[str], message: str) -> None:
    print(
        f"Warning: failed to fully clean process group {getattr(proc, 'pid', '?')}: {message}",
        file=sys.stderr,
    )


def process_group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_process_group_exit(
    pid: int,
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while process_group_alive(pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_seconds, remaining))
    return True


def stop_process(proc: subprocess.Popen[str] | None, *, timeout_seconds: float = 30.0) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as exc:  # pragma: no cover - defensive cleanup path.
        raise RuntimeError(f"SIGTERM failed for process group {proc.pid}") from exc

    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        pass
    except Exception as exc:  # pragma: no cover - defensive cleanup path.
        raise RuntimeError(f"failed waiting for process leader {proc.pid}") from exc

    if not process_group_alive(proc.pid):
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as exc:  # pragma: no cover - defensive cleanup path.
        raise RuntimeError(
            f"SIGKILL failed for process group {proc.pid}"
        ) from exc
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        pass
    except Exception as exc:  # pragma: no cover - defensive cleanup path.
        raise RuntimeError(
            f"failed reaping process leader {proc.pid}"
        ) from exc
    if not wait_for_process_group_exit(
        proc.pid,
        timeout_seconds=timeout_seconds,
    ):
        raise RuntimeError(
            f"process group {proc.pid} remained alive after SIGKILL"
        )


def cleanup_processes(processes: Iterable[subprocess.Popen[str] | None]) -> None:
    for process in processes:
        try:
            stop_process(process)
        except Exception as exc:  # pragma: no cover - last-resort cleanup isolation.
            if process is not None:
                _cleanup_warning(process, str(exc))


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


def wait_for_forwarded_process_exit(
    process: subprocess.Popen[str] | None,
    *,
    timeout_seconds: float,
) -> bool:
    if process is None or process.poll() is not None:
        return True
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


class ForwardedSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def install_process_group_signal_handlers(
    processes: Callable[[], Iterable[subprocess.Popen[str] | None]],
) -> dict[int, object]:
    previous_handlers: dict[int, object] = {}
    shutdown_initiated = False
    handler_active = False
    followup_forwarded = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal shutdown_initiated, handler_active, followup_forwarded
        if handler_active or followup_forwarded:
            return
        is_followup = shutdown_initiated
        shutdown_initiated = True
        handler_active = True
        try:
            if is_followup:
                followup_forwarded = True
                mask_process_group_signals_for_cleanup()
            forward_signal_to_live_process_groups(signum, processes())
        finally:
            handler_active = False
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
    managed_signals = {signal.SIGTERM, signal.SIGHUP, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, managed_signals)
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


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


def read_log_tail(log_path: Path, *, max_bytes: int = 16_384) -> str:
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            return log_file.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<unable to read log: {exc}>"


def raise_if_server_exited(
    process: subprocess.Popen[str], *, role: str, log_path: Path
) -> None:
    exit_code = process.poll()
    if exit_code is None:
        return
    log_tail = read_log_tail(log_path)
    raise ServerExitedBeforeHealthy(
        f"vLLM server {role!r} exited before becoming healthy "
        f"with exit code {exit_code}; log: {log_path}\n"
        f"--- log tail ---\n{log_tail}"
    )


def response_serves_model(response: object, expected_model_name: str) -> bool:
    if int(getattr(response, "status_code", 0)) != 200:
        return False
    try:
        payload = response.json()  # type: ignore[attr-defined]
    except (TypeError, ValueError, AttributeError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return False
    return any(
        isinstance(item, dict) and item.get("id") == expected_model_name
        for item in payload["data"]
    )


def wait_for_server(
    base_url: str,
    *,
    process: subprocess.Popen[str],
    role: str,
    log_path: Path,
    expected_model_name: str,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
) -> None:
    if requests is None:
        raise RuntimeError("requests is required for vLLM health checks")
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        raise_if_server_exited(process, role=role, log_path=log_path)
        try:
            response = requests.get(f"{base_url}/models", timeout=10.0)
            raise_if_server_exited(process, role=role, log_path=log_path)
            if response_serves_model(response, expected_model_name):
                return
            last_error = RuntimeError(
                f"health response did not serve expected model {expected_model_name!r}"
            )
        except Exception as exc:  # pragma: no cover - integration only.
            if isinstance(exc, ServerExitedBeforeHealthy):
                raise
            last_error = exc
        time.sleep(poll_seconds)
    raise_if_server_exited(process, role=role, log_path=log_path)
    raise RuntimeError(f"Timed out waiting for {base_url}/models: {last_error}")


def start_and_wait_server(
    spec: VllmServerSpec,
    *,
    runtime_dir: Path,
    timeout_seconds: float,
) -> subprocess.Popen[str]:
    for attempt in (1, 2):
        log_path = runtime_dir / f"{spec.role}.attempt-{attempt}.log"
        process = start_server(spec, log_path=log_path)
        try:
            wait_for_server(
                spec.base_url,
                process=process,
                role=spec.role,
                log_path=log_path,
                expected_model_name=spec.served_model_name,
                timeout_seconds=timeout_seconds,
            )
            return process
        except ServerExitedBeforeHealthy:
            stop_process(process)
            if attempt == 2:
                raise
        except BaseException:
            stop_process(process)
            raise
    raise AssertionError("unreachable")


def write_endpoint_file(path: Path, urls: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        value = url.strip().rstrip("/")
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(deduped) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_ready_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "vllm_servers_ready", "timestamp": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_json_marker(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def round_event_path(control_dir: Path, round_id: int, event: str) -> Path:
    return control_dir / f"round-{round_id:06d}.{event}.json"


@dataclass
class RoundServiceProcesses:
    text: subprocess.Popen[str] | None
    primary_image: subprocess.Popen[str] | None
    secondary_image: subprocess.Popen[str] | None


def stop_round_services(
    processes: RoundServiceProcesses,
    *,
    text_endpoints_file: Path,
    image_endpoints_file: Path,
) -> None:
    """Withdraw every EntiTables endpoint before returning the local GPUs."""
    endpoint_error: BaseException | None = None
    try:
        write_endpoint_file(text_endpoints_file, [])
        write_endpoint_file(image_endpoints_file, [])
    except BaseException as error:
        endpoint_error = error
    finally:
        process_errors: list[BaseException] = []
        for process in (
            processes.text,
            processes.primary_image,
            processes.secondary_image,
        ):
            try:
                stop_process(process)
            except BaseException as error:
                process_errors.append(error)
        processes.text = None
        processes.primary_image = None
        processes.secondary_image = None
    if endpoint_error is not None:
        raise endpoint_error
    if process_errors:
        raise RuntimeError(
            "failed to stop every EntiTables vLLM process group"
        ) from process_errors[0]


def start_round_services(
    *,
    round_id: int,
    text_task_count: int,
    image_task_count: int,
    runtime_dir: Path,
    text_server: VllmServerSpec,
    primary_image_server: VllmServerSpec,
    secondary_image_server: VllmServerSpec,
    text_endpoints_file: Path,
    image_endpoints_file: Path,
    processes: RoundServiceProcesses,
    server_start_timeout_seconds: float,
    gpu_priority_owner: PriorityGpuOwner | None,
) -> None:
    """Reclaim GPUs and start only services needed by one model round."""
    if processes.text or processes.primary_image or processes.secondary_image:
        raise RuntimeError("prior EntiTables model services were not released")
    write_endpoint_file(text_endpoints_file, [])
    write_endpoint_file(image_endpoints_file, [])
    if text_task_count <= 0 and image_task_count <= 0:
        return
    if gpu_priority_owner is not None:
        gpu_priority_owner.request_gpus(reason=f"model_round_{round_id}")
    try:
        if image_task_count > 0:
            processes.primary_image = start_and_wait_server(
                primary_image_server,
                runtime_dir=runtime_dir,
                timeout_seconds=server_start_timeout_seconds,
            )
        if text_task_count > 0:
            processes.text = start_and_wait_server(
                text_server,
                runtime_dir=runtime_dir,
                timeout_seconds=server_start_timeout_seconds,
            )
        elif image_task_count > 0:
            processes.secondary_image = start_and_wait_server(
                secondary_image_server,
                runtime_dir=runtime_dir,
                timeout_seconds=server_start_timeout_seconds,
            )
        write_endpoint_file(
            text_endpoints_file,
            [text_server.base_url] if processes.text is not None else [],
        )
        image_urls = []
        if processes.primary_image is not None:
            image_urls.append(primary_image_server.base_url)
        if processes.secondary_image is not None:
            image_urls.append(secondary_image_server.base_url)
        write_endpoint_file(image_endpoints_file, image_urls)
    except BaseException:
        stop_round_services(
            processes,
            text_endpoints_file=text_endpoints_file,
            image_endpoints_file=image_endpoints_file,
        )
        if gpu_priority_owner is not None:
            gpu_priority_owner.release_gpus(
                reason=f"model_round_{round_id}_startup_failed"
            )
        raise


def _run_round_service_loop_owned(
    *,
    builder: subprocess.Popen[str],
    control_dir: Path,
    run_id: str,
    runtime_dir: Path,
    text_server: VllmServerSpec,
    primary_image_server: VllmServerSpec,
    secondary_image_server: VllmServerSpec,
    text_endpoints_file: Path,
    image_endpoints_file: Path,
    processes: RoundServiceProcesses,
    server_start_timeout_seconds: float,
    round_timeout_seconds: float | None,
    gpu_priority_owner: PriorityGpuOwner | None,
    poll_seconds: float = 0.2,
) -> int:
    round_id = 0
    active_start: dict[str, object] | None = None
    last_activity = time.time()
    while True:
        exit_code = builder.poll()
        if exit_code is not None:
            return int(exit_code)
        if (
            active_start is not None
            and round_timeout_seconds is not None
            and time.time() - last_activity > round_timeout_seconds
        ):
            raise RuntimeError(f"Timed out waiting for dynamic model round {round_id}")

        start_path = round_event_path(control_dir, round_id, "start")
        if active_start is None:
            payload = read_json_marker(start_path)
            if (
                payload is None
                or payload.get("status") != "model_round_start"
                or payload.get("round_id") != round_id
                or payload.get("run_id") != run_id
            ):
                time.sleep(poll_seconds)
                continue
            active_start = payload
            last_activity = time.time()
            text_task_count = max(0, int(payload.get("text_task_count", 0)))
            image_task_count = max(
                0,
                int(payload.get("image_task_count", 0)),
            )
            start_round_services(
                round_id=round_id,
                text_task_count=text_task_count,
                image_task_count=image_task_count,
                runtime_dir=runtime_dir,
                text_server=text_server,
                primary_image_server=primary_image_server,
                secondary_image_server=secondary_image_server,
                text_endpoints_file=text_endpoints_file,
                image_endpoints_file=image_endpoints_file,
                processes=processes,
                server_start_timeout_seconds=server_start_timeout_seconds,
                gpu_priority_owner=gpu_priority_owner,
            )
            write_atomic_json(
                round_event_path(control_dir, round_id, "ready"),
                {
                    "status": "model_round_services_ready",
                    "round_id": round_id,
                    "run_id": run_id,
                    "timestamp": time.time(),
                },
            )

        done_path = round_event_path(control_dir, round_id, "done")
        done_payload = read_json_marker(done_path)
        round_done = bool(
            done_payload
            and done_payload.get("round_id") == round_id
            and done_payload.get("run_id") == run_id
            and done_payload.get("status")
            in {"model_round_completed", "model_round_failed"}
        )
        if round_done:
            stop_round_services(
                processes,
                text_endpoints_file=text_endpoints_file,
                image_endpoints_file=image_endpoints_file,
            )
            if gpu_priority_owner is not None:
                gpu_priority_owner.release_gpus(
                    reason=f"after_model_round_{round_id}"
                )
            active_start = None
            round_id += 1
            last_activity = time.time()
            continue

        text_done_path = round_event_path(control_dir, round_id, "text.done")
        image_done_path = round_event_path(control_dir, round_id, "image.done")
        text_done_payload = read_json_marker(text_done_path)
        image_done_payload = read_json_marker(image_done_path)
        text_done = bool(
            text_done_payload
            and text_done_payload.get("round_id") == round_id
            and text_done_payload.get("run_id") == run_id
            and text_done_payload.get("status") == "text_round_tasks_completed"
        )
        image_done = bool(
            image_done_payload
            and image_done_payload.get("round_id") == round_id
            and image_done_payload.get("run_id") == run_id
            and image_done_payload.get("status") == "image_round_tasks_completed"
        )
        text_task_count = max(0, int(active_start.get("text_task_count", 0)))
        image_task_count = max(0, int(active_start.get("image_task_count", 0)))
        if (
            image_task_count > 0
            and processes.text is not None
            and (text_task_count == 0 or text_done)
            and not image_done
            and not round_done
        ):
            # The text future has completed, so clearing its authoritative
            # endpoint snapshot and stopping the service cannot interrupt work.
            write_endpoint_file(text_endpoints_file, [])
            stop_process(processes.text)
            processes.text = None
            processes.secondary_image = start_and_wait_server(
                secondary_image_server,
                runtime_dir=runtime_dir,
                timeout_seconds=server_start_timeout_seconds,
            )
            image_done_payload = read_json_marker(image_done_path)
            done_payload = read_json_marker(done_path)
            image_finished_while_starting = bool(
                image_done_payload
                and image_done_payload.get("round_id") == round_id
                and image_done_payload.get("run_id") == run_id
            ) or bool(
                done_payload
                and done_payload.get("round_id") == round_id
                and done_payload.get("run_id") == run_id
            )
            if image_finished_while_starting:
                stop_process(processes.secondary_image)
                processes.secondary_image = None
            else:
                write_endpoint_file(
                    image_endpoints_file,
                    [primary_image_server.base_url, secondary_image_server.base_url],
                )
            last_activity = time.time()
        time.sleep(poll_seconds)


def run_round_service_loop(
    *,
    builder: subprocess.Popen[str],
    control_dir: Path,
    run_id: str,
    runtime_dir: Path,
    text_server: VllmServerSpec,
    primary_image_server: VllmServerSpec,
    secondary_image_server: VllmServerSpec,
    text_endpoints_file: Path,
    image_endpoints_file: Path,
    text_process: subprocess.Popen[str] | None,
    primary_image_process: subprocess.Popen[str] | None,
    secondary_image_process: subprocess.Popen[str] | None,
    server_start_timeout_seconds: float,
    round_timeout_seconds: float | None,
    gpu_priority_owner: PriorityGpuOwner | None = None,
    poll_seconds: float = 0.2,
) -> int:
    processes = RoundServiceProcesses(
        text=text_process,
        primary_image=primary_image_process,
        secondary_image=secondary_image_process,
    )
    try:
        return _run_round_service_loop_owned(
            builder=builder,
            control_dir=control_dir,
            run_id=run_id,
            runtime_dir=runtime_dir,
            text_server=text_server,
            primary_image_server=primary_image_server,
            secondary_image_server=secondary_image_server,
            text_endpoints_file=text_endpoints_file,
            image_endpoints_file=image_endpoints_file,
            processes=processes,
            server_start_timeout_seconds=server_start_timeout_seconds,
            round_timeout_seconds=round_timeout_seconds,
            gpu_priority_owner=gpu_priority_owner,
            poll_seconds=poll_seconds,
        )
    finally:
        stop_round_services(
            processes,
            text_endpoints_file=text_endpoints_file,
            image_endpoints_file=image_endpoints_file,
        )
        if gpu_priority_owner is not None:
            gpu_priority_owner.release_gpus(reason="round_service_loop_stopped")


def read_pending_model_task_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if "runner_startup_task_count" in payload:
        try:
            return max(0, int(payload["runner_startup_task_count"]))
        except (TypeError, ValueError):
            return None
    if payload.get("round_mode") is True:
        return None
    if not {
        "text_task_count",
        "image_task_count",
    }.issubset(payload):
        return None
    try:
        return max(0, int(payload["text_task_count"])) + max(0, int(payload["image_task_count"]))
    except (TypeError, ValueError):
        return None


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


def with_model_workers(
    passthrough_args: list[str],
    *,
    text_workers: int,
    image_workers: int,
) -> list[str]:
    result = list(passthrough_args)
    for option, workers in (
        ("--text_model_workers", text_workers),
        ("--image_model_workers", image_workers),
    ):
        if workers <= 0:
            continue
        if not passthrough_has_arg(result, option):
            result.extend([option, str(workers)])
    return result


def with_remote_model_workers(
    passthrough_args: list[str],
    *,
    text_workers: int,
    image_workers: int,
) -> list[str]:
    result = list(passthrough_args)
    for option, workers in (
        ("--remote_text_model_workers", text_workers),
        ("--remote_image_model_workers", image_workers),
    ):
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
    parser.add_argument(
        "--forwarded_signal_grace_seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--model_start_timeout_seconds", type=float, default=None, help="Maximum seconds to wait for the builder to finish Wikipedia/material preparation before vLLM startup. Default waits indefinitely.")
    parser.add_argument("--first_done_timeout_seconds", type=float, default=None)
    parser.add_argument("--text_done_timeout_seconds", type=float, default=None, help="Deprecated alias for --first_done_timeout_seconds.")
    parser.add_argument(
        "--text_model_workers",
        type=int,
        default=2,
        help="Concurrent local text requests; 0 leaves the builder default unchanged.",
    )
    parser.add_argument(
        "--image_model_workers",
        type=int,
        default=2,
        help="Concurrent local image requests; 0 leaves the builder default unchanged.",
    )
    parser.add_argument(
        "--gpu_coordination_dir",
        default=None,
        help=(
            "Optional shared directory for lending both local GPUs to a WDC "
            "borrower during EntiTables network-only phases."
        ),
    )
    parser.add_argument(
        "--gpu_reclaim_timeout_seconds",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--gpu_borrower_stale_seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--gpu_unregistered_grace_seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gpu_coordination_poll_seconds",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--remote_layout_control_url",
        default=None,
        help=(
            "Optional loopback control tunnel used to borrow WDC's remote "
            "GPUs while its priority request is borrowable."
        ),
    )
    parser.add_argument("--remote_layout_control_token_file")
    parser.add_argument("--remote_layout_primary_image_url")
    parser.add_argument("--remote_layout_switchable_url")
    parser.add_argument("--remote_layout_coordination_dir")
    parser.add_argument(
        "--remote_text_model_workers",
        type=int,
        default=0,
        help="Total concurrent remote text requests across routed text endpoints.",
    )
    parser.add_argument(
        "--remote_image_model_workers",
        type=int,
        default=0,
        help="Total concurrent remote image requests across routed image endpoints.",
    )
    parser.add_argument(
        "--remote_layout_lease_ttl_seconds", type=int, default=15
    )
    parser.add_argument(
        "--remote_layout_lease_renew_seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--remote_layout_request_timeout_seconds", type=float, default=3.0
    )
    parser.add_argument(
        "--remote_layout_reconnect_timeout_seconds", type=float, default=30.0
    )
    parser.add_argument(
        "--remote_layout_operation_timeout_seconds", type=float, default=1200.0
    )
    parser.add_argument(
        "--remote_layout_drain_timeout_seconds", type=float, default=300.0
    )
    parser.add_argument("--remote_layout_poll_seconds", type=float, default=1.0)
    parser.add_argument(
        "--remote_layout_workload_poll_seconds", type=float, default=2.0
    )
    parser.add_argument(
        "--remote_layout_stability_seconds", type=float, default=0.0
    )
    parser.add_argument(
        "--remote_layout_health_timeout_seconds", type=float, default=3.0
    )
    parser.add_argument(
        "--remote_layout_health_stable_polls", type=int, default=2
    )
    parser.add_argument(
        "--remote_layout_coordination_poll_seconds", type=float, default=0.2
    )
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
    args, passthrough = parser.parse_known_args(argv)
    if args.gpu_coordination_dir and (
        args.gpu_reclaim_timeout_seconds <= 0
        or args.gpu_borrower_stale_seconds <= 0
        or args.gpu_unregistered_grace_seconds < 0
        or args.gpu_coordination_poll_seconds <= 0
    ):
        parser.error("GPU coordination timeouts must be positive")
    remote_required = (
        args.remote_layout_control_token_file,
        args.remote_layout_primary_image_url,
        args.remote_layout_switchable_url,
        args.remote_layout_coordination_dir,
    )
    if args.remote_layout_control_url and not all(remote_required):
        parser.error(
            "remote layout control requires token, both inference tunnels, "
            "and the shared WDC coordination directory"
        )
    if not args.remote_layout_control_url and any(remote_required):
        parser.error(
            "remote layout settings require --remote_layout_control_url"
        )
    if passthrough_has_arg(passthrough, "--dynamic_model_workers"):
        parser.error(
            "--dynamic_model_workers was replaced by "
            "--text_model_workers and --image_model_workers"
        )
    if passthrough_has_arg(passthrough, "--remote_dynamic_model_workers"):
        parser.error(
            "--remote_dynamic_model_workers was replaced by "
            "--remote_text_model_workers and --remote_image_model_workers"
        )
    if min(args.text_model_workers, args.image_model_workers) < 0:
        parser.error("local model worker counts must be non-negative")
    if min(
        args.remote_text_model_workers,
        args.remote_image_model_workers,
    ) < 0:
        parser.error("remote model worker counts must be non-negative")
    if (
        args.remote_layout_control_url
        and args.remote_text_model_workers == 0
        and args.remote_image_model_workers == 0
    ):
        parser.error(
            "remote layout control requires at least one positive remote "
            "model worker count"
        )
    if args.remote_layout_control_url:
        if not 5 <= args.remote_layout_lease_ttl_seconds <= 60:
            parser.error("remote layout lease TTL must be between 5 and 60")
        if not (
            0
            < args.remote_layout_lease_renew_seconds
            < args.remote_layout_lease_ttl_seconds
        ):
            parser.error("remote layout renewal must be positive and below TTL")
        positive = (
            args.remote_layout_request_timeout_seconds,
            args.remote_layout_reconnect_timeout_seconds,
            args.remote_layout_operation_timeout_seconds,
            args.remote_layout_drain_timeout_seconds,
            args.remote_layout_poll_seconds,
            args.remote_layout_workload_poll_seconds,
            args.remote_layout_health_timeout_seconds,
            args.remote_layout_coordination_poll_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            parser.error("remote layout timeouts must be finite and positive")
        if (
            not math.isfinite(args.remote_layout_stability_seconds)
            or args.remote_layout_stability_seconds < 0
        ):
            parser.error("remote layout stability must be non-negative")
        if args.remote_layout_health_stable_polls <= 0:
            parser.error("remote layout stable polls must be positive")
    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough_args = parse_args(argv)
    if not (
        math.isfinite(args.forwarded_signal_grace_seconds)
        and args.forwarded_signal_grace_seconds >= 0
    ):
        raise ValueError(
            "--forwarded_signal_grace_seconds must be finite and non-negative"
        )
    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "_dynamic_vllm"
    text_endpoints_file = runtime_dir / "text_endpoints.txt"
    image_endpoints_file = runtime_dir / "image_endpoints.txt"
    model_start_marker = runtime_dir / "model_start.json"
    model_ready_marker = runtime_dir / "model_ready.json"
    text_done_marker = runtime_dir / "text_done.json"
    image_done_marker = runtime_dir / "image_done.json"
    model_round_control_dir = runtime_dir / "rounds"
    model_round_run_id = uuid.uuid4().hex
    remote_routing_manifest = runtime_dir / "remote_model_routing.json"
    remote_controller_id_file = runtime_dir / "remote_layout_controller_id"
    remote_controller_lock_file = runtime_dir / "remote_layout_controller.lock"
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
    gpu_priority_owner = (
        PriorityGpuOwner(
            Path(args.gpu_coordination_dir),
            gpu_ids=(
                args.primary_image_gpu,
                args.text_gpu,
                args.secondary_image_gpu,
            ),
            reclaim_timeout_seconds=args.gpu_reclaim_timeout_seconds,
            borrower_stale_seconds=args.gpu_borrower_stale_seconds,
            unregistered_grace_seconds=(
                args.gpu_unregistered_grace_seconds
            ),
            poll_seconds=args.gpu_coordination_poll_seconds,
        )
        if args.gpu_coordination_dir
        else None
    )

    text_proc: subprocess.Popen[str] | None = None
    primary_image_proc: subprocess.Popen[str] | None = None
    secondary_text_proc: subprocess.Popen[str] | None = None
    secondary_image_proc: subprocess.Popen[str] | None = None
    builder_proc: subprocess.Popen[str] | None = None
    round_loop_manages_priority = False
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
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for marker in (model_start_marker, model_ready_marker, text_done_marker, image_done_marker):
            if marker.exists():
                marker.unlink()
        model_round_control_dir.mkdir(parents=True, exist_ok=True)
        for marker in model_round_control_dir.glob("round-*.json*"):
            marker.unlink()
        write_endpoint_file(text_endpoints_file, [])
        write_endpoint_file(image_endpoints_file, [])
        if gpu_priority_owner is not None:
            gpu_priority_owner.release_gpus(
                reason="entitables_network_preparation"
            )

        builder_passthrough_args = with_model_workers(
            passthrough_args,
            text_workers=args.text_model_workers,
            image_workers=args.image_model_workers,
        )
        remote_layout_args: list[str] = []
        if args.remote_layout_control_url:
            builder_passthrough_args = with_remote_model_workers(
                builder_passthrough_args,
                text_workers=args.remote_text_model_workers,
                image_workers=args.remote_image_model_workers,
            )
            remote_layout_args = [
                "--remote_model_routing_manifest",
                str(remote_routing_manifest),
                "--remote_layout_control_url",
                args.remote_layout_control_url,
                "--remote_layout_control_token_file",
                args.remote_layout_control_token_file,
                "--remote_layout_controller_id_file",
                str(remote_controller_id_file),
                "--remote_layout_lock_file",
                str(remote_controller_lock_file),
                "--remote_layout_primary_image_url",
                args.remote_layout_primary_image_url,
                "--remote_layout_switchable_url",
                args.remote_layout_switchable_url,
                "--remote_layout_coordination_dir",
                args.remote_layout_coordination_dir,
                "--remote_layout_lease_ttl_seconds",
                str(args.remote_layout_lease_ttl_seconds),
                "--remote_layout_lease_renew_seconds",
                str(args.remote_layout_lease_renew_seconds),
                "--remote_layout_request_timeout_seconds",
                str(args.remote_layout_request_timeout_seconds),
                "--remote_layout_reconnect_timeout_seconds",
                str(args.remote_layout_reconnect_timeout_seconds),
                "--remote_layout_operation_timeout_seconds",
                str(args.remote_layout_operation_timeout_seconds),
                "--remote_layout_drain_timeout_seconds",
                str(args.remote_layout_drain_timeout_seconds),
                "--remote_layout_poll_seconds",
                str(args.remote_layout_poll_seconds),
                "--remote_layout_workload_poll_seconds",
                str(args.remote_layout_workload_poll_seconds),
                "--remote_layout_stability_seconds",
                str(args.remote_layout_stability_seconds),
                "--remote_layout_health_timeout_seconds",
                str(args.remote_layout_health_timeout_seconds),
                "--remote_layout_health_stable_polls",
                str(args.remote_layout_health_stable_polls),
                "--remote_layout_coordination_poll_seconds",
                str(args.remote_layout_coordination_poll_seconds),
            ]
        builder_command = build_builder_command(
            python_executable=args.python_executable,
            builder_script=Path(args.builder_script),
            input_dir=Path(args.input_dir),
            output_dir=output_dir,
            text_server=text_server,
            primary_image_server=primary_image_server,
            text_endpoints_file=text_endpoints_file,
            image_endpoints_file=image_endpoints_file,
            model_start_marker=model_start_marker,
            model_ready_marker=model_ready_marker,
            text_done_marker=text_done_marker,
            image_done_marker=image_done_marker,
            model_round_control_dir=model_round_control_dir,
            model_round_run_id=model_round_run_id,
            passthrough_args=builder_passthrough_args,
            remote_layout_args=remote_layout_args,
        )
        builder_proc = subprocess.Popen(builder_command, text=True, start_new_session=True)
        wait_for_marker_or_builder_exit(
            marker=model_start_marker,
            builder=builder_proc,
            timeout_seconds=args.model_start_timeout_seconds,
        )

        start_payload = read_json_marker(model_start_marker) or {}
        if start_payload.get("round_mode") is True:
            write_ready_marker(model_ready_marker)
            if (
                read_pending_model_task_count(model_start_marker) == 0
                or (
                    text_done_marker.exists()
                    and image_done_marker.exists()
                )
            ):
                return int(builder_proc.wait())
            round_loop_manages_priority = True
            return run_round_service_loop(
                builder=builder_proc,
                control_dir=model_round_control_dir,
                run_id=model_round_run_id,
                runtime_dir=runtime_dir,
                text_server=text_server,
                primary_image_server=primary_image_server,
                secondary_image_server=secondary_image_server,
                text_endpoints_file=text_endpoints_file,
                image_endpoints_file=image_endpoints_file,
                text_process=None,
                primary_image_process=None,
                secondary_image_process=None,
                server_start_timeout_seconds=args.server_start_timeout_seconds,
                round_timeout_seconds=first_done_timeout,
                gpu_priority_owner=gpu_priority_owner,
            )

        if read_pending_model_task_count(model_start_marker) == 0:
            write_ready_marker(model_ready_marker)
            return int(builder_proc.wait())

        if gpu_priority_owner is not None:
            gpu_priority_owner.request_gpus(reason="single_model_phase")
        text_proc = start_and_wait_server(
            text_server,
            runtime_dir=runtime_dir,
            timeout_seconds=args.server_start_timeout_seconds,
        )
        primary_image_proc = start_and_wait_server(
            primary_image_server,
            runtime_dir=runtime_dir,
            timeout_seconds=args.server_start_timeout_seconds,
        )
        write_endpoint_file(text_endpoints_file, [text_server.base_url])
        write_endpoint_file(
            image_endpoints_file,
            [primary_image_server.base_url],
        )
        write_ready_marker(model_ready_marker)

        completed = wait_for_any_marker_or_builder_exit(
            markers={"text": text_done_marker, "image": image_done_marker},
            builder=builder_proc,
            timeout_seconds=first_done_timeout,
        )

        if completed == {"text"} and not image_done_marker.exists():
            stop_process(text_proc)
            text_proc = None
            secondary_image_proc = start_and_wait_server(
                secondary_image_server,
                runtime_dir=runtime_dir,
                timeout_seconds=args.server_start_timeout_seconds,
            )
            write_endpoint_file(image_endpoints_file, [primary_image_server.base_url, secondary_image_server.base_url])

        return int(builder_proc.wait())
    except ForwardedSignal as exc:
        try:
            try:
                wait_for_forwarded_process_exit(
                    builder_proc,
                    timeout_seconds=args.forwarded_signal_grace_seconds,
                )
            except ForwardedSignal:
                pass
        finally:
            mask_process_group_signals_for_cleanup()
        return 128 + exc.signum
    finally:
        mask_process_group_signals_for_cleanup()
        try:
            cleanup_processes([builder_proc, secondary_text_proc])
            cleanup_succeeded = True
            try:
                stop_round_services(
                    RoundServiceProcesses(
                        text=text_proc,
                        primary_image=primary_image_proc,
                        secondary_image=secondary_image_proc,
                    ),
                    text_endpoints_file=text_endpoints_file,
                    image_endpoints_file=image_endpoints_file,
                )
            except BaseException as error:
                cleanup_succeeded = False
                print(
                    f"Warning: EntiTables model cleanup was incomplete: {error}",
                    file=sys.stderr,
                )
            if (
                gpu_priority_owner is not None
                and not round_loop_manages_priority
                and cleanup_succeeded
            ):
                gpu_priority_owner.release_gpus(
                    reason="entitables_runner_stopped"
                )
        finally:
            restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
