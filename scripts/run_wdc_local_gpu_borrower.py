#!/usr/bin/env python
"""Lend idle EntiTables GPUs to WDC through preemptible local vLLM servers.

The WDC builder keeps its remote endpoints as the fixed base URLs and reads
the two endpoint files managed here before every request.  On an EntiTables
reclaim request this sidecar first empties those files, then terminates both
local vLLM process groups, and only then acknowledges the matching request.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import gpu_priority_protocol as protocol

try:
    import requests
except ImportError:  # pragma: no cover - integration environment issue.
    requests = None  # type: ignore[assignment]


BALANCED_LAYOUT = "text-image"
IMAGE_ONLY_LAYOUT = "image-image"


@dataclass(frozen=True)
class VllmServerSpec:
    role: str
    model_path: str
    served_model_name: str
    gpu: str
    port: int
    extra_args: tuple[str, ...]
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


@dataclass(frozen=True)
class WorkloadDemand:
    text: int
    image: int


def read_workload_demand(database_path: Path) -> WorkloadDemand:
    """Read unfinished work for the newest durable WDC model job-set pair."""
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro",
        uri=True,
        timeout=1.0,
    )
    try:
        pair = connection.execute(
            """
            SELECT text_kind, image_kind
            FROM model_jobset_pairs
            ORDER BY updated_at DESC, identity DESC
            LIMIT 1
            """
        ).fetchone()
        if pair is None:
            return WorkloadDemand(text=0, image=0)
        text_kind, image_kind = (str(pair[0]), str(pair[1]))
        counts = {text_kind: 0, image_kind: 0}
        rows = connection.execute(
            """
            SELECT kind, COUNT(*)
            FROM jobs
            WHERE kind IN (?, ?)
              AND status NOT IN ('success', 'terminal')
            GROUP BY kind
            """,
            (text_kind, image_kind),
        ).fetchall()
        for kind, count in rows:
            counts[str(kind)] = int(count)
        return WorkloadDemand(
            text=counts[text_kind],
            image=counts[image_kind],
        )
    finally:
        connection.close()


def _clean_arg(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def default_vllm_args(args: argparse.Namespace) -> list[str]:
    if args.no_default_vllm_memory_args:
        return []
    return [
        "--trust-remote-code",
        "--dtype",
        _clean_arg(args.vllm_dtype),
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--gpu-memory-utilization",
        _clean_arg(args.vllm_gpu_memory_utilization),
        "--enforce-eager",
        "--skip-mm-profiling",
        "--mm-processor-cache-gb",
        _clean_arg(args.vllm_mm_processor_cache_gb),
        "--max-num-batched-tokens",
        str(args.vllm_max_num_batched_tokens),
        "--max-num-seqs",
        str(args.vllm_max_num_seqs),
    ]


def process_env(gpu: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    return environment


def start_server(
    spec: VllmServerSpec,
    *,
    log_dir: Path,
    attempt: int,
) -> subprocess.Popen[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.role}.attempt-{attempt:04d}.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(
            spec.command(),
            env=process_env(spec.gpu),
            text=True,
            start_new_session=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )


def process_group_alive(process: subprocess.Popen[str]) -> bool:
    # Reap an exited direct child before probing its process group.  Until the
    # leader is reaped, killpg(..., 0) still succeeds for a zombie-only group
    # and can make a completed SIGTERM/SIGKILL look like a failed shutdown.
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


class BorrowerController:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        text_server: VllmServerSpec,
        image_server: VllmServerSpec,
        secondary_image_server: VllmServerSpec,
    ) -> None:
        self.args = args
        self.paths = protocol.ProtocolPaths(
            Path(args.coordination_dir).resolve()
        )
        self.text_endpoints_file = Path(
            args.text_endpoints_file
        ).resolve()
        self.image_endpoints_file = Path(
            args.image_endpoints_file
        ).resolve()
        self.text_server = text_server
        self.image_server = image_server
        self.secondary_image_server = secondary_image_server
        self.model_jobs_database = (
            Path(args.model_jobs_database).resolve()
            if args.model_jobs_database
            else None
        )
        self.borrower_id = uuid.uuid4().hex
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self._active_layout: str | None = None
        self._desired_layout = BALANCED_LAYOUT
        self._last_workload_poll = 0.0
        self.shutdown_requested = False
        self._last_heartbeat = 0.0
        self._attempt = 0
        self._last_released_token: tuple[str, int] | None = None
        self._last_serving_token: tuple[str, int] | None = None
        self._api_key = args.api_key or os.getenv("VLLM_API_KEY")

    def request_shutdown(self, _signum: int, _frame: object) -> None:
        self.shutdown_requested = True

    def heartbeat(
        self,
        state: str,
        request: protocol.PriorityRequest | None,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat < self.args.heartbeat_seconds:
            return
        protocol.write_borrower_status(
            self.paths.borrower,
            borrower_id=self.borrower_id,
            state=state,
            request=request,
        )
        self._last_heartbeat = now

    def clear_endpoints(self) -> None:
        protocol.atomic_write_endpoints(self.text_endpoints_file, [])
        protocol.atomic_write_endpoints(self.image_endpoints_file, [])

    def desired_layout(self, *, force: bool = False) -> str:
        if self.model_jobs_database is None:
            return BALANCED_LAYOUT
        now = time.monotonic()
        if (
            not force
            and now - self._last_workload_poll
            < self.args.workload_poll_seconds
        ):
            return self._desired_layout
        self._last_workload_poll = now
        try:
            demand = read_workload_demand(self.model_jobs_database)
        except (OSError, sqlite3.Error, ValueError):
            logging.exception(
                "Failed to read WDC model workload from %s; keeping %s",
                self.model_jobs_database,
                self._desired_layout,
            )
            return self._desired_layout
        desired = (
            IMAGE_ONLY_LAYOUT
            if demand.text == 0 and demand.image > 0
            else BALANCED_LAYOUT
        )
        if desired != self._desired_layout:
            logging.info(
                "WDC workload changed: text=%d image=%d layout=%s",
                demand.text,
                demand.image,
                desired,
            )
        self._desired_layout = desired
        return desired

    def _layout_specs(
        self,
        layout: str,
    ) -> dict[str, VllmServerSpec]:
        if layout == BALANCED_LAYOUT:
            return {
                self.text_server.role: self.text_server,
                self.image_server.role: self.image_server,
            }
        if layout == IMAGE_ONLY_LAYOUT:
            return {
                self.image_server.role: self.image_server,
                self.secondary_image_server.role: (
                    self.secondary_image_server
                ),
            }
        raise ValueError(f"unsupported WDC borrower layout: {layout}")

    def _publish_layout_endpoints(self, layout: str) -> None:
        if layout == BALANCED_LAYOUT:
            text_urls = [self.text_server.base_url]
            image_urls = [self.image_server.base_url]
        elif layout == IMAGE_ONLY_LAYOUT:
            text_urls = []
            image_urls = [
                self.image_server.base_url,
                self.secondary_image_server.base_url,
            ]
        else:
            raise ValueError(f"unsupported WDC borrower layout: {layout}")
        protocol.atomic_write_endpoints(
            self.text_endpoints_file,
            text_urls,
        )
        protocol.atomic_write_endpoints(
            self.image_endpoints_file,
            image_urls,
        )

    def _request_is_current(
        self,
        request: protocol.PriorityRequest,
    ) -> bool:
        return protocol.request_is_current(self.paths.request, request)

    def _request_allows_servers(
        self,
        request: protocol.PriorityRequest,
    ) -> bool:
        return (
            request.state == protocol.BORROWABLE_STATE
            and {self.text_server.gpu, self.image_server.gpu}
            <= set(request.gpu_ids)
        )

    def _probe_server(self, spec: VllmServerSpec) -> bool:
        if requests is None:
            raise RuntimeError("requests is required for vLLM health checks")
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = requests.get(
                f"{spec.base_url}/models",
                headers=headers,
                timeout=self.args.health_request_timeout_seconds,
            )
            payload = response.json()
        except Exception:
            return False
        if (
            int(getattr(response, "status_code", 0)) != 200
            or not isinstance(payload, dict)
            or not isinstance(payload.get("data"), list)
        ):
            return False
        return any(
            isinstance(item, dict)
            and item.get("id") == spec.served_model_name
            for item in payload["data"]
        )

    def _start_servers(
        self,
        request: protocol.PriorityRequest,
        *,
        layout: str | None = None,
    ) -> bool:
        layout = layout or self.desired_layout()
        specs = self._layout_specs(layout)
        self.clear_endpoints()
        if not self._request_is_current(request):
            return False
        self._attempt += 1
        self.processes = {}
        try:
            for role, spec in specs.items():
                self.processes[role] = start_server(
                    spec,
                    log_dir=self.paths.root / "logs",
                    attempt=self._attempt,
                )
        except BaseException:
            logging.exception("Failed to launch WDC borrower services")
            self.stop_servers(request=request, state="startup_failed")
            return False
        ready: set[str] = set()
        deadline = time.monotonic() + self.args.server_start_timeout_seconds
        try:
            while len(ready) < len(specs):
                self.heartbeat("starting", request)
                if (
                    self.shutdown_requested
                    or not self._request_is_current(request)
                    or not self._request_allows_servers(request)
                ):
                    return False
                for role, spec in specs.items():
                    if role in ready:
                        continue
                    process = self.processes[role]
                    exit_code = process.poll()
                    if exit_code is not None:
                        raise RuntimeError(
                            f"{role} borrower vLLM exited with code {exit_code}; "
                            f"see {self.paths.root / 'logs'}"
                        )
                    if self._probe_server(spec):
                        ready.add(role)
                if len(ready) == len(specs):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "timed out starting WDC borrower vLLM servers"
                    )
                time.sleep(self.args.poll_seconds)
            if not self._request_is_current(request):
                return False
            self._publish_layout_endpoints(layout)
            if not self._request_is_current(request):
                self.clear_endpoints()
                return False
            protocol.write_acknowledgement(
                self.paths.acknowledgement,
                request,
                borrower_id=self.borrower_id,
                status=protocol.SERVING_STATUS,
            )
            self._last_serving_token = request.token
            self._active_layout = layout
            self.heartbeat("serving", request, force=True)
            logging.info(
                "WDC now borrowing GPUs %s with layout=%s",
                ",".join(request.gpu_ids),
                layout,
            )
            return True
        except Exception:
            logging.exception("Failed to start WDC borrower services")
            return False

    def _signal_groups(self, signum: int) -> None:
        for process in self.processes.values():
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                continue

    def _wait_for_groups(
        self,
        timeout_seconds: float,
        *,
        request: protocol.PriorityRequest | None,
        state: str,
    ) -> list[subprocess.Popen[str]]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            alive = [
                process
                for process in self.processes.values()
                if process_group_alive(process)
            ]
            if not alive or time.monotonic() >= deadline:
                return alive
            self.heartbeat(state, request)
            time.sleep(self.args.poll_seconds)

    def stop_servers(
        self,
        *,
        request: protocol.PriorityRequest | None,
        state: str,
    ) -> None:
        endpoint_error: BaseException | None = None
        try:
            self.clear_endpoints()
        except BaseException as error:
            endpoint_error = error
        if not self.processes:
            if endpoint_error is not None:
                raise endpoint_error
            return
        self._signal_groups(signal.SIGTERM)
        alive = self._wait_for_groups(
            self.args.stop_timeout_seconds,
            request=request,
            state=state,
        )
        if alive:
            for process in alive:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
            alive = self._wait_for_groups(
                self.args.kill_timeout_seconds,
                request=request,
                state=state,
            )
        for process in self.processes.values():
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        self.processes.clear()
        self._last_serving_token = None
        self._active_layout = None
        if alive:
            pids = ", ".join(str(process.pid) for process in alive)
            raise RuntimeError(
                f"WDC borrower process groups did not stop: {pids}"
            )
        if endpoint_error is not None:
            raise endpoint_error

    def _acknowledge_release(
        self,
        request: protocol.PriorityRequest,
    ) -> None:
        self.stop_servers(request=request, state="releasing")
        if not self._request_is_current(request):
            return
        protocol.write_acknowledgement(
            self.paths.acknowledgement,
            request,
            borrower_id=self.borrower_id,
            status=protocol.RELEASED_STATUS,
        )
        self._last_released_token = request.token
        self.heartbeat("released", request, force=True)
        logging.info(
            "Returned GPUs to EntiTables for generation=%s sequence=%d",
            request.generation,
            request.sequence,
        )

    def _sleep_while_current(
        self,
        request: protocol.PriorityRequest,
        seconds: float,
        *,
        state: str,
    ) -> None:
        deadline = time.monotonic() + seconds
        while (
            not self.shutdown_requested
            and self._request_is_current(request)
            and time.monotonic() < deadline
        ):
            self.heartbeat(state, request)
            time.sleep(
                min(
                    self.args.poll_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )

    def run(self) -> int:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.clear_endpoints()
        self.heartbeat("waiting", None, force=True)
        while not self.shutdown_requested:
            request = protocol.read_priority_request(self.paths.request)
            if request is None:
                self.stop_servers(request=None, state="waiting")
                self.heartbeat("waiting", None)
                time.sleep(self.args.poll_seconds)
                continue
            if request.state == protocol.PRIORITY_REQUESTED_STATE:
                if (
                    request.token != self._last_released_token
                    or self.processes
                ):
                    self._acknowledge_release(request)
                else:
                    self.heartbeat("released", request)
                time.sleep(self.args.poll_seconds)
                continue
            if not self._request_allows_servers(request):
                self.stop_servers(request=request, state="waiting")
                self.heartbeat("waiting", request)
                time.sleep(self.args.poll_seconds)
                continue
            desired_layout = self.desired_layout()
            if (
                self.processes
                and self._active_layout != desired_layout
            ):
                logging.info(
                    "Reconfiguring WDC borrower from %s to %s",
                    self._active_layout,
                    desired_layout,
                )
                self.stop_servers(
                    request=request,
                    state="reconfiguring",
                )
            if not self.processes:
                if not self._start_servers(
                    request,
                    layout=desired_layout,
                ):
                    self.stop_servers(request=request, state="retrying")
                    self._sleep_while_current(
                        request,
                        self.args.restart_backoff_seconds,
                        state="retrying",
                    )
                    continue
            dead = [
                role
                for role, process in self.processes.items()
                if process.poll() is not None
            ]
            if dead:
                logging.error(
                    "Borrower service exited unexpectedly: %s",
                    ", ".join(dead),
                )
                self.stop_servers(request=request, state="retrying")
                self._sleep_while_current(
                    request,
                    self.args.restart_backoff_seconds,
                    state="retrying",
                )
                continue
            self.heartbeat("serving", request)
            time.sleep(self.args.poll_seconds)
        return 0

    def close(self) -> None:
        try:
            self.stop_servers(request=None, state="stopping")
        finally:
            self.clear_endpoints()
            protocol.remove_borrower_status(
                self.paths.borrower,
                borrower_id=self.borrower_id,
            )


def _acquire_singleton_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(
            f"another WDC GPU borrower already owns {path}"
        ) from None
    return handle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preemptible local WDC vLLM endpoints while EntiTables is "
            "in network-only phases."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--coordination_dir", required=True)
    parser.add_argument("--text_endpoints_file", required=True)
    parser.add_argument("--image_endpoints_file", required=True)
    parser.add_argument("--text_model_path", required=True)
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_gpu", default="1")
    parser.add_argument("--text_port", type=int, default=18101)
    parser.add_argument("--image_model_path", required=True)
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Thinking")
    parser.add_argument("--image_gpu", default="0")
    parser.add_argument("--image_port", type=int, default=18100)
    parser.add_argument("--secondary_image_port", type=int, default=18102)
    parser.add_argument(
        "--model_jobs_database",
        help=(
            "Optional WDC model jobs SQLite database. When its newest text "
            "queue is complete but image work remains, both GPUs serve the "
            "image model."
        ),
    )
    parser.add_argument("--workload_poll_seconds", type=float, default=5.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--vllm_bin", default="vllm")
    parser.add_argument("--api_key")
    parser.add_argument("--server_start_timeout_seconds", type=float, default=900.0)
    parser.add_argument("--health_request_timeout_seconds", type=float, default=2.0)
    parser.add_argument("--stop_timeout_seconds", type=float, default=20.0)
    parser.add_argument("--kill_timeout_seconds", type=float, default=10.0)
    parser.add_argument("--heartbeat_seconds", type=float, default=1.0)
    parser.add_argument("--poll_seconds", type=float, default=0.2)
    parser.add_argument("--restart_backoff_seconds", type=float, default=5.0)
    parser.add_argument("--vllm_dtype", default="bfloat16")
    parser.add_argument("--vllm_max_model_len", type=int, default=8192)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=2048)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=4)
    parser.add_argument("--vllm_mm_processor_cache_gb", type=float, default=0)
    parser.add_argument("--no_default_vllm_memory_args", action="store_true")
    parser.add_argument("--vllm_extra_arg", action="append", default=[])
    parser.add_argument("--text_vllm_extra_arg", action="append", default=[])
    parser.add_argument("--image_vllm_extra_arg", action="append", default=[])
    args = parser.parse_args(argv)
    positive = (
        args.server_start_timeout_seconds,
        args.health_request_timeout_seconds,
        args.stop_timeout_seconds,
        args.kill_timeout_seconds,
        args.heartbeat_seconds,
        args.poll_seconds,
        args.restart_backoff_seconds,
        args.workload_poll_seconds,
    )
    if any(value <= 0 for value in positive):
        parser.error("all timeout, heartbeat, poll, and backoff values must be positive")
    if len(
        {
            args.text_port,
            args.image_port,
            args.secondary_image_port,
        }
    ) != 3:
        parser.error("borrower ports must be distinct")
    if (
        Path(args.text_endpoints_file).resolve()
        == Path(args.image_endpoints_file).resolve()
    ):
        parser.error("text and image endpoint files must differ")
    return args


def _server_specs(
    args: argparse.Namespace,
) -> tuple[VllmServerSpec, VllmServerSpec, VllmServerSpec]:
    common = [*default_vllm_args(args), *args.vllm_extra_arg]
    return (
        VllmServerSpec(
            role="wdc-text",
            model_path=args.text_model_path,
            served_model_name=args.text_model_name,
            gpu=args.text_gpu,
            port=args.text_port,
            host=args.host,
            vllm_bin=args.vllm_bin,
            extra_args=tuple([*common, *args.text_vllm_extra_arg]),
        ),
        VllmServerSpec(
            role="wdc-image",
            model_path=args.image_model_path,
            served_model_name=args.image_model_name,
            gpu=args.image_gpu,
            port=args.image_port,
            host=args.host,
            vllm_bin=args.vllm_bin,
            extra_args=tuple([*common, *args.image_vllm_extra_arg]),
        ),
        VllmServerSpec(
            role="wdc-image-secondary",
            model_path=args.image_model_path,
            served_model_name=args.image_model_name,
            gpu=args.text_gpu,
            port=args.secondary_image_port,
            host=args.host,
            vllm_bin=args.vllm_bin,
            extra_args=tuple([*common, *args.image_vllm_extra_arg]),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    text_server, image_server, secondary_image_server = _server_specs(args)
    controller = BorrowerController(
        args,
        text_server=text_server,
        image_server=image_server,
        secondary_image_server=secondary_image_server,
    )
    lock_handle = _acquire_singleton_lock(controller.paths.borrower_lock)
    previous_handlers = {
        signum: signal.signal(signum, controller.request_shutdown)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        return controller.run()
    finally:
        controller.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
