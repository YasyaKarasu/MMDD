from __future__ import annotations

import json
import socket
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

from remote_vllm_layout import (
    ControllerConfig,
    EndpointConfig,
    EndpointVerificationError,
    ExclusiveControllerLock,
    LayoutProtocolError,
    RemoteLayoutController,
    RemoteTransportError,
    RoutedEndpoint,
    RoutingManifest,
    RoutingScheduler,
    RoutingUnavailableError,
    WorkloadSnapshot,
    desired_layout_for_workload,
    load_routing_manifest,
    read_durable_workload,
)


TEXT_MODEL = "Qwen3.5-9B"
IMAGE_MODEL = "Qwen3-VL-8B-Thinking"
TOKEN = "fake-layout-token"


class FakeModelState:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


@contextmanager
def fake_model_endpoint(state: FakeModelState) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self.send_error(404)
                return
            body = json.dumps({"data": [{"id": state.model_id}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class FakeAgentState:
    def __init__(self, layout: str = "balanced") -> None:
        self.lock = threading.Lock()
        self.boot_id = str(uuid.uuid4())
        self.layout = layout
        self.lease_id = str(uuid.uuid4())
        self.controller_id: str | None = None
        self.session_id: str | None = None
        self.operations: dict[str, dict[str, Any]] = {}
        self.operation_by_key: dict[tuple[str, str, int], str] = {}
        self.pending_targets: dict[str, tuple[str, str]] = {}
        self.layout_request_count = 0
        self.lease_acquire_count = 0
        self.lease_history: list[tuple[str, str]] = []
        self.operation_get_history: list[str] = []
        self.drop_layout_response_once = False
        self.drop_post_operation_status_once = False
        self.restart_before_operation_response_once = False
        self.fail_next_operation = False
        self.rollback_next_operation = False
        self.switch_endpoint_state: FakeModelState | None = None
        self.switch_generation = 8

    def services(self) -> dict[str, dict[str, Any]]:
        switch_role = "text" if self.layout == "balanced" else "image"
        switch_model = TEXT_MODEL if switch_role == "text" else IMAGE_MODEL
        switch_capacity = 128 if switch_role == "text" else 32
        return {
            "primary_image": {
                "endpoint_id": "primary_image",
                "role": "image",
                "state": "ready",
                "remote_port": 8000,
                "served_model_id": IMAGE_MODEL,
                "instance_generation": 3,
                "max_inflight": 32,
            },
            "switchable": {
                "endpoint_id": "switchable",
                "role": switch_role,
                "state": "ready",
                "remote_port": 8001,
                "served_model_id": switch_model,
                "instance_generation": self.switch_generation,
                "max_inflight": switch_capacity,
            },
        }


@contextmanager
def fake_agent(state: FakeAgentState) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length)) if length else {}

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if self.headers.get("Authorization") == f"Bearer {TOKEN}":
                return True
            self._send(
                401,
                {
                    "schema_version": "mmdd-remote-vllm-layout-v1",
                    "agent_boot_id": state.boot_id,
                    "error": {
                        "code": "unauthorized",
                        "message": "unauthorized",
                        "retryable": False,
                    },
                },
            )
            return False

        def do_PUT(self) -> None:
            if not self._authorized():
                return
            request = self._body()
            if self.path == "/v1/lease":
                with state.lock:
                    state.lease_acquire_count += 1
                    state.lease_history.append((state.boot_id, state.lease_id))
                    state.controller_id = request["controller_id"]
                    state.session_id = request["session_id"]
                    response = {
                        "schema_version": "mmdd-remote-vllm-layout-v1",
                        "agent_boot_id": state.boot_id,
                        "lease_id": state.lease_id,
                        "controller_id": state.controller_id,
                        "session_id": state.session_id,
                        "ttl_seconds": request["requested_ttl_seconds"],
                        "server_time_unix_ms": 1,
                    }
                self._send(200, response)
                return
            if self.path != "/v1/layout":
                self.send_error(404)
                return
            key = (
                request["controller_id"],
                request["session_id"],
                request["sequence"],
            )
            with state.lock:
                state.layout_request_count += 1
                operation_id = state.operation_by_key.get(key)
                if operation_id is None:
                    operation_id = str(uuid.uuid4())
                    prior = state.layout
                    operation_state = (
                        "failed" if state.fail_next_operation else "accepted"
                    )
                    terminal_state = (
                        "rolled_back"
                        if state.rollback_next_operation
                        else "ready"
                    )
                    operation = {
                        "operation_id": operation_id,
                        "controller_id": request["controller_id"],
                        "session_id": request["session_id"],
                        "sequence": request["sequence"],
                        "from_layout": prior,
                        "desired_layout": request["desired_layout"],
                        "resulting_layout": (
                            state.layout if operation_state == "ready" else None
                        ),
                        "state": operation_state,
                        "error_code": (
                            None if operation_state == "ready" else "start_failed"
                        ),
                        "error_message": (
                            None if operation_state == "ready" else "fake failure"
                        ),
                        "services": (
                            state.services() if operation_state == "ready" else None
                        ),
                    }
                    state.operations[operation_id] = operation
                    state.operation_by_key[key] = operation_id
                    if operation_state == "accepted":
                        state.pending_targets[operation_id] = (
                            request["desired_layout"],
                            terminal_state,
                        )
                    state.fail_next_operation = False
                    state.rollback_next_operation = False
                operation = dict(state.operations[operation_id])
                drop = state.drop_layout_response_once
                state.drop_layout_response_once = False
            if drop:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self._send(
                202 if operation["state"] != "ready" else 200,
                {
                    "schema_version": "mmdd-remote-vllm-layout-v1",
                    "agent_boot_id": state.boot_id,
                    "operation": operation,
                },
            )

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path == "/v1/status":
                with state.lock:
                    if (
                        state.drop_post_operation_status_once
                        and state.layout_request_count > 0
                        and not state.pending_targets
                    ):
                        state.drop_post_operation_status_once = False
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                        return
                    response = {
                        "schema_version": "mmdd-remote-vllm-layout-v1",
                        "agent_boot_id": state.boot_id,
                        "agent_state": "ready",
                        "ready_layout": state.layout,
                        "target_layout": state.layout,
                        "active_operation_id": None,
                        "status_revision": 1,
                        "services": state.services(),
                        "server_time_unix_ms": 1,
                    }
                self._send(200, response)
                return
            prefix = "/v1/operations/"
            if self.path.startswith(prefix):
                operation_id = self.path[len(prefix) :]
                with state.lock:
                    state.operation_get_history.append(operation_id)
                    if state.restart_before_operation_response_once:
                        state.restart_before_operation_response_once = False
                        state.boot_id = str(uuid.uuid4())
                        state.lease_id = str(uuid.uuid4())
                        operation = dict(state.operations[operation_id])
                        self._send(
                            200,
                            {
                                "schema_version": "mmdd-remote-vllm-layout-v1",
                                "agent_boot_id": state.boot_id,
                                "operation": operation,
                            },
                        )
                        return
                    pending = state.pending_targets.pop(operation_id, None)
                    if pending is not None:
                        target, terminal_state = pending
                        if terminal_state == "ready":
                            if state.layout != target:
                                state.switch_generation += 1
                            state.layout = target
                            if state.switch_endpoint_state is not None:
                                state.switch_endpoint_state.model_id = (
                                    TEXT_MODEL
                                    if target == "balanced"
                                    else IMAGE_MODEL
                                )
                        state.operations[operation_id].update(
                            state=terminal_state,
                            resulting_layout=state.layout,
                            services=state.services(),
                        )
                    operation = dict(state.operations[operation_id])
                self._send(
                    200,
                    {
                        "schema_version": "mmdd-remote-vllm-layout-v1",
                        "agent_boot_id": state.boot_id,
                        "operation": operation,
                    },
                )
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if not self._authorized():
                return
            self._body()
            self._send(
                200,
                {
                    "schema_version": "mmdd-remote-vllm-layout-v1",
                    "agent_boot_id": state.boot_id,
                },
            )

        def log_message(self, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def create_database(
    path: Path,
    *,
    text_statuses: tuple[str, ...],
    image_statuses: tuple[str, ...],
    identity: str = "pair-1",
    updated_at: float = 1.0,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_jobset_pairs (
                identity TEXT PRIMARY KEY,
                text_kind TEXT NOT NULL,
                image_kind TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO model_jobset_pairs VALUES (?, 'text-kind', 'image-kind', ?)",
            (identity, updated_at),
        )
        for index, status in enumerate(text_statuses):
            connection.execute(
                "INSERT INTO jobs VALUES (?, 'text-kind', ?)",
                (f"{identity}-text-{index}", status),
            )
        for index, status in enumerate(image_statuses):
            connection.execute(
                "INSERT INTO jobs VALUES (?, 'image-kind', ?)",
                (f"{identity}-image-{index}", status),
            )


def controller_config(
    tmp_path: Path,
    *,
    control_url: str,
    database_path: Path,
    primary_url: str,
    switchable_url: str,
) -> ControllerConfig:
    token = tmp_path / "layout-token"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    token.chmod(0o600)
    return ControllerConfig(
        control_url=control_url,
        token_file=token,
        database_path=database_path,
        controller_id_file=tmp_path / "controller-id",
        lock_file=tmp_path / "controller.lock",
        endpoints={
            "primary_image": EndpointConfig("primary_image", primary_url),
            "switchable": EndpointConfig("switchable", switchable_url),
        },
        text_model_id=TEXT_MODEL,
        image_model_id=IMAGE_MODEL,
        lease_ttl_seconds=5,
        lease_renew_seconds=1.0,
        request_timeout_seconds=0.2,
        reconnect_timeout_seconds=2.0,
        operation_timeout_seconds=2.0,
        drain_timeout_seconds=2.0,
        poll_seconds=0.01,
        workload_poll_seconds=60.0,
        image_burst_stability_seconds=0.0,
        endpoint_health_timeout_seconds=0.2,
        endpoint_health_stable_polls=1,
    )


def make_manifest(
    *,
    revision: int,
    layout: str = "balanced",
) -> RoutingManifest:
    primary = RoutedEndpoint(
        "primary_image", "http://127.0.0.1:18000/v1", IMAGE_MODEL, 1, 32
    )
    switch = RoutedEndpoint(
        "switchable",
        "http://127.0.0.1:18001/v1",
        TEXT_MODEL if layout == "balanced" else IMAGE_MODEL,
        2,
        128 if layout == "balanced" else 32,
    )
    return RoutingManifest(
        controller_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        routing_revision=revision,
        agent_boot_id=str(uuid.uuid4()),
        remote_operation_id=None,
        layout=layout,
        text=(switch,) if layout == "balanced" else (),
        image=(primary,) if layout == "balanced" else (primary, switch),
    )


def test_durable_workload_uses_newest_pair_and_all_nonterminal_statuses(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("success", "terminal"),
        image_statuses=("pending",),
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO model_jobset_pairs VALUES ('pair-2', 'new-text', 'new-image', 2)"
        )
        connection.executemany(
            "INSERT INTO jobs VALUES (?, ?, ?)",
            [
                ("new-text-1", "new-text", "leased"),
                ("new-text-2", "new-text", "unexpected-status"),
                ("new-image-1", "new-image", "success"),
            ],
        )

    workload = read_durable_workload(database)

    assert workload.pair_identity == "pair-2"
    assert workload.text_unfinished == 2
    assert workload.image_unfinished == 0
    assert desired_layout_for_workload(
        workload, current_healthy_layout="image_burst"
    ) == "balanced"


def test_controller_identity_persists_sessions_rotate_and_lock_is_exclusive(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(database, text_statuses=(), image_statuses=())
    config = controller_config(
        tmp_path,
        control_url="http://127.0.0.1:18999",
        database_path=database,
        primary_url="http://127.0.0.1:18000/v1",
        switchable_url="http://127.0.0.1:18001/v1",
    )
    first = RemoteLayoutController(
        config, RoutingScheduler(tmp_path / "routing-1.json")
    )
    second = RemoteLayoutController(
        config, RoutingScheduler(tmp_path / "routing-2.json")
    )

    assert first.controller_id == second.controller_id
    assert first.session_id != second.session_id
    primary_lock = ExclusiveControllerLock(config.lock_file)
    competing_lock = ExclusiveControllerLock(config.lock_file)
    primary_lock.acquire()
    try:
        with pytest.raises(LayoutProtocolError, match="another local"):
            competing_lock.acquire()
    finally:
        primary_lock.release()


def test_scheduler_withdrawal_is_atomic_with_leases_and_capacity(
    tmp_path: Path,
) -> None:
    scheduler = RoutingScheduler(tmp_path / "routing.json")
    balanced = make_manifest(revision=1)
    scheduler.publish(balanced)
    leased = scheduler.acquire("text")
    withdrawn = RoutingManifest(
        controller_id=balanced.controller_id,
        session_id=balanced.session_id,
        routing_revision=2,
        agent_boot_id=balanced.agent_boot_id,
        remote_operation_id=None,
        layout="balanced",
        text=(),
        image=balanced.image,
    )
    scheduler.publish(withdrawn)

    assert scheduler.inflight("switchable") == 1
    assert scheduler.wait_drained("switchable", 0.01) is False
    with pytest.raises(RoutingUnavailableError):
        scheduler.acquire("text")
    scheduler.release(leased)
    assert scheduler.wait_drained("switchable", 0.1) is True


def test_scheduler_clear_withdraws_persisted_routes_and_waits_for_all_drains(
    tmp_path: Path,
) -> None:
    routing_path = tmp_path / "routing.json"
    scheduler = RoutingScheduler(routing_path)
    scheduler.publish(make_manifest(revision=1))
    leased = scheduler.acquire("text")

    scheduler.clear()

    assert scheduler.snapshot() is None
    assert not routing_path.exists()
    assert scheduler.endpoints("text") == ()
    assert scheduler.wait_all_drained(0.01) is False
    scheduler.release(leased)
    assert scheduler.wait_all_drained(0.1) is True


def test_controller_can_read_entitables_in_memory_workload(tmp_path: Path) -> None:
    expected = WorkloadSnapshot(
        pair_identity="round-1",
        text_kind="entitables-text",
        image_kind="entitables-image",
        text_unfinished=3,
        image_unfinished=5,
    )
    config = controller_config(
        tmp_path,
        control_url="http://127.0.0.1:18999",
        database_path=tmp_path / "not-created.sqlite3",
        primary_url="http://127.0.0.1:18000/v1",
        switchable_url="http://127.0.0.1:18001/v1",
    )
    config = replace(
        config,
        database_path=None,
        workload_reader=lambda: expected,
    )
    controller = RemoteLayoutController(
        config,
        RoutingScheduler(tmp_path / "routing-memory.json"),
    )

    assert controller._read_workload() == expected


def test_scheduler_enforces_each_capacity_and_round_robin_ties(
    tmp_path: Path,
) -> None:
    scheduler = RoutingScheduler(tmp_path / "routing.json")
    base = make_manifest(revision=1, layout="image_burst")
    primary = RoutedEndpoint(
        "primary_image", base.image[0].base_url, IMAGE_MODEL, 1, 1
    )
    switch = RoutedEndpoint(
        "switchable", base.image[1].base_url, IMAGE_MODEL, 2, 1
    )
    scheduler.publish(
        RoutingManifest(
            controller_id=base.controller_id,
            session_id=base.session_id,
            routing_revision=1,
            agent_boot_id=base.agent_boot_id,
            remote_operation_id=None,
            layout="image_burst",
            text=(),
            image=(primary, switch),
        )
    )
    first = scheduler.acquire("image")
    second = scheduler.acquire("image")
    acquired: list[RoutedEndpoint] = []
    started = threading.Event()

    def wait_for_capacity() -> None:
        started.set()
        acquired.append(scheduler.acquire("image"))

    thread = threading.Thread(target=wait_for_capacity)
    thread.start()
    assert started.wait(1.0)
    thread.join(0.02)
    assert thread.is_alive()
    assert [first.endpoint_id, second.endpoint_id] == [
        "primary_image",
        "switchable",
    ]
    scheduler.release(first)
    thread.join(1.0)
    assert acquired == [primary]
    scheduler.release(second)
    scheduler.release(acquired[0])


def test_controller_balanced_uses_published_text_capacity_128(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("pending",),
        image_statuses=("pending",),
    )
    agent_state = FakeAgentState(layout="balanced")
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(FakeModelState(TEXT_MODEL)) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert manifest.layout == "balanced"
            assert scheduler.capacity("text") == 128
            assert scheduler.capacity("image") == 32
            assert manifest.text[0].max_inflight == 128

            leased = [scheduler.acquire("text") for _ in range(128)]
            acquired: list[RoutedEndpoint] = []
            acquired_event = threading.Event()

            def acquire_after_capacity() -> None:
                acquired.append(scheduler.acquire("text"))
                acquired_event.set()

            thread = threading.Thread(target=acquire_after_capacity)
            thread.start()
            assert not acquired_event.wait(0.03)
            scheduler.release(leased[0])
            assert acquired_event.wait(1.0)
            thread.join(1.0)
            for endpoint in leased[1:]:
                scheduler.release(endpoint)
            scheduler.release(acquired[0])
            assert scheduler.inflight("switchable") == 0


def test_controller_switches_to_image_burst_and_publishes_capacity_64(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("success",),
        image_statuses=("pending", "leased"),
    )
    primary_state = FakeModelState(IMAGE_MODEL)
    switch_state = FakeModelState(TEXT_MODEL)
    agent_state = FakeAgentState(layout="balanced")
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(primary_state) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert manifest.layout == "image_burst"
            assert scheduler.capacity("text") == 0
            assert scheduler.capacity("image") == 64
            assert [item.max_inflight for item in manifest.image] == [32, 32]
            assert [item.endpoint_id for item in manifest.image] == [
                "primary_image",
                "switchable",
            ]
            assert manifest.image[1].instance_generation == 9
            assert load_routing_manifest(tmp_path / "routing.json") == manifest


def test_rolled_back_transition_restores_text_role_and_capacity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("success",),
        image_statuses=("pending",),
    )
    switch_state = FakeModelState(TEXT_MODEL)
    agent_state = FakeAgentState(layout="balanced")
    agent_state.rollback_next_operation = True
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert agent_state.layout_request_count == 1
            assert manifest.layout == "balanced"
            assert scheduler.capacity("text") == 128
            assert scheduler.capacity("image") == 32
            assert manifest.text[0].served_model_id == TEXT_MODEL
            assert manifest.text[0].instance_generation == 8


@pytest.mark.parametrize(
    ("rolled_back", "expected_layout", "text_capacity", "image_capacity"),
    [
        (False, "image_burst", 0, 64),
        (True, "balanced", 128, 32),
    ],
)
def test_terminal_operation_snapshot_rebuilds_routes_before_status_reconnect(
    tmp_path: Path,
    rolled_back: bool,
    expected_layout: str,
    text_capacity: int,
    image_capacity: int,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("pending",),
        image_statuses=("pending",),
    )
    switch_state = FakeModelState(TEXT_MODEL)
    agent_state = FakeAgentState(layout="balanced")
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE jobs SET status = 'success' WHERE kind = 'text-kind'"
                )
            agent_state.rollback_next_operation = rolled_back
            agent_state.drop_post_operation_status_once = True

            with pytest.raises(RemoteTransportError):
                controller.reconcile_once()

            manifest = scheduler.snapshot()
            assert manifest is not None
            assert manifest.layout == expected_layout
            assert scheduler.capacity("text") == text_capacity
            assert scheduler.capacity("image") == image_capacity
            if expected_layout == "image_burst":
                assert [endpoint.max_inflight for endpoint in manifest.image] == [
                    32,
                    32,
                ]
            else:
                assert manifest.text[0].max_inflight == 128


def test_transition_withdraws_before_drain_and_existing_lease_finishes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("pending",),
        image_statuses=("pending",),
    )
    switch_state = FakeModelState(TEXT_MODEL)
    agent_state = FakeAgentState(layout="balanced")
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            request_started = threading.Event()
            allow_request_to_finish = threading.Event()
            request_finished = threading.Event()

            def existing_request() -> None:
                try:
                    with scheduler.lease("text"):
                        request_started.set()
                        allow_request_to_finish.wait(2.0)
                finally:
                    request_finished.set()

            request_thread = threading.Thread(target=existing_request)
            request_thread.start()
            assert request_started.wait(1.0)
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE jobs SET status = 'success' WHERE kind = 'text-kind'"
                )

            errors: list[BaseException] = []

            def reconcile() -> None:
                try:
                    controller.reconcile_once()
                except BaseException as error:
                    errors.append(error)

            transition_thread = threading.Thread(target=reconcile)
            transition_thread.start()
            deadline = time.monotonic() + 1.0
            while scheduler.capacity("text") != 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            try:
                assert scheduler.capacity("text") == 0
                assert scheduler.inflight("switchable") == 1
                assert agent_state.layout_request_count == 0
                with pytest.raises(RoutingUnavailableError):
                    scheduler.acquire("text")
            finally:
                allow_request_to_finish.set()
            request_thread.join(1.0)
            transition_thread.join(2.0)

            assert request_finished.is_set()
            assert not request_thread.is_alive()
            assert not transition_thread.is_alive()
            assert errors == []
            assert scheduler.inflight("switchable") == 0
            assert scheduler.capacity("text") == 0
            assert scheduler.capacity("image") == 64
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert [endpoint.max_inflight for endpoint in manifest.image] == [
                32,
                32,
            ]


def test_lost_layout_response_retries_same_idempotency_key(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("success",),
        image_statuses=("pending",),
    )
    agent_state = FakeAgentState(layout="balanced")
    agent_state.drop_layout_response_once = True
    switch_state = FakeModelState(TEXT_MODEL)
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        with RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        ):
            pass

    assert agent_state.layout_request_count == 2
    assert len(agent_state.operation_by_key) == 1
    assert scheduler.capacity("image") == 64


def test_wrong_tunnel_model_never_publishes_route(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("pending",),
        image_statuses=("pending",),
    )
    agent_state = FakeAgentState(layout="balanced")
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(FakeModelState("wrong-model")) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with pytest.raises(EndpointVerificationError):
            controller.start()

    assert scheduler.capacity("text") == 0
    assert scheduler.capacity("image") == 0
    assert not (tmp_path / "routing.json").exists()


def test_agent_restart_withdraws_old_route_and_reconciles_new_boot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("pending",),
        image_statuses=("pending",),
    )
    agent_state = FakeAgentState(layout="balanced")
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(FakeModelState(TEXT_MODEL)) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            prior_boot = controller.agent_boot_id
            prior_lease = controller._lease_snapshot().lease_id
            prior_acquisitions = agent_state.lease_acquire_count
            with agent_state.lock:
                agent_state.boot_id = str(uuid.uuid4())
                agent_state.lease_id = str(uuid.uuid4())
            controller.reconcile_once()
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert manifest.agent_boot_id == agent_state.boot_id
            assert manifest.agent_boot_id != prior_boot
            assert controller._lease_snapshot().lease_id == agent_state.lease_id
            assert controller._lease_snapshot().lease_id != prior_lease
            assert agent_state.lease_acquire_count > prior_acquisitions
            assert agent_state.lease_history[-1] == (
                agent_state.boot_id,
                agent_state.lease_id,
            )
            assert scheduler.capacity("text") == 128
            assert scheduler.capacity("image") == 32


def test_agent_restart_invalidates_inflight_operation_context(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("pending",),
        image_statuses=("pending",),
    )
    switch_state = FakeModelState(TEXT_MODEL)
    agent_state = FakeAgentState(layout="balanced")
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            prior_boot = controller.agent_boot_id
            prior_lease = controller._lease_snapshot().lease_id
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE jobs SET status = 'success' WHERE kind = 'text-kind'"
                )
            agent_state.restart_before_operation_response_once = True

            controller.reconcile_once()

            assert controller.agent_boot_id == agent_state.boot_id
            assert controller.agent_boot_id != prior_boot
            assert controller._lease_snapshot().lease_id == agent_state.lease_id
            assert controller._lease_snapshot().lease_id != prior_lease
            assert len(agent_state.operation_get_history) == 2
            assert len(set(agent_state.operation_get_history)) == 2
            assert len(agent_state.operation_by_key) == 2
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert manifest.agent_boot_id == agent_state.boot_id
            assert manifest.layout == "image_burst"
            assert scheduler.capacity("text") == 0
            assert scheduler.capacity("image") == 64


def test_new_jobset_generation_with_text_returns_to_balanced(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("success",),
        image_statuses=("pending",),
    )
    switch_state = FakeModelState(IMAGE_MODEL)
    agent_state = FakeAgentState(layout="image_burst")
    agent_state.switch_endpoint_state = switch_state
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(switch_state) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with controller:
            assert scheduler.capacity("image") == 64
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "INSERT INTO model_jobset_pairs VALUES ('pair-2', 'new-text', 'new-image', 2)"
                )
                connection.executemany(
                    "INSERT INTO jobs VALUES (?, ?, ?)",
                    [
                        ("new-text-1", "new-text", "pending"),
                        ("new-image-1", "new-image", "pending"),
                    ],
                )
            controller.reconcile_once()
            manifest = scheduler.snapshot()
            assert manifest is not None
            assert manifest.layout == "balanced"
            assert scheduler.capacity("text") == 128
            assert scheduler.capacity("image") == 32
            assert manifest.text[0].max_inflight == 128
            assert manifest.text[0].instance_generation == 9


def test_failed_operation_keeps_switchable_withdrawn(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    create_database(
        database,
        text_statuses=("success",),
        image_statuses=("pending",),
    )
    initial = make_manifest(revision=1, layout="balanced")
    agent_state = FakeAgentState(layout="balanced")
    agent_state.fail_next_operation = True
    with (
        fake_model_endpoint(FakeModelState(IMAGE_MODEL)) as primary_url,
        fake_model_endpoint(FakeModelState(TEXT_MODEL)) as switch_url,
        fake_agent(agent_state) as control_url,
    ):
        scheduler = RoutingScheduler(tmp_path / "routing.json")
        scheduler.publish(initial)
        controller = RemoteLayoutController(
            controller_config(
                tmp_path,
                control_url=control_url,
                database_path=database,
                primary_url=primary_url,
                switchable_url=switch_url,
            ),
            scheduler,
        )
        with pytest.raises(Exception, match="ended in failed"):
            controller.start()

    manifest = scheduler.snapshot()
    assert manifest is not None
    assert manifest.text == ()
    assert [item.endpoint_id for item in manifest.image] == ["primary_image"]
