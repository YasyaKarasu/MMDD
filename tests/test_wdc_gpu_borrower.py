import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

import gpu_priority_protocol as protocol
import run_wdc_local_gpu_borrower as borrower


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        self.alive = False
        return 0


def make_controller(
    tmp_path: Path,
    *,
    extra_args: tuple[str, ...] = (),
) -> borrower.BorrowerController:
    args = borrower.parse_args(
        [
            "--coordination_dir",
            str(tmp_path / "coordination"),
            "--text_endpoints_file",
            str(tmp_path / "wdc-text.txt"),
            "--image_endpoints_file",
            str(tmp_path / "wdc-image.txt"),
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/image",
            "--server_start_timeout_seconds",
            "2",
            "--health_request_timeout_seconds",
            "0.1",
            "--stop_timeout_seconds",
            "0.1",
            "--kill_timeout_seconds",
            "0.1",
            "--heartbeat_seconds",
            "0.01",
            "--poll_seconds",
            "0.01",
            "--restart_backoff_seconds",
            "0.01",
            *extra_args,
        ]
    )
    text_server, image_server, secondary_image_server = (
        borrower._server_specs(args)
    )
    return borrower.BorrowerController(
        args,
        text_server=text_server,
        image_server=image_server,
        secondary_image_server=secondary_image_server,
    )


def publish_request(
    controller: borrower.BorrowerController,
    *,
    state: str,
    sequence: int = 1,
) -> protocol.PriorityRequest:
    request = protocol.PriorityRequest(
        generation="generation-1",
        sequence=sequence,
        state=state,
        owner="entitables",
        gpu_ids=("0", "1"),
        reason="test",
        timestamp=1.0,
    )
    protocol.atomic_write_json(
        controller.paths.request,
        request.payload(),
    )
    return request


def test_borrower_publishes_local_endpoints_only_after_both_servers_are_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = make_controller(tmp_path)
    request = publish_request(
        controller,
        state=protocol.BORROWABLE_STATE,
    )
    processes = iter((FakeProcess(101), FakeProcess(102)))
    monkeypatch.setattr(
        borrower,
        "start_server",
        lambda *_args, **_kwargs: next(processes),
    )
    monkeypatch.setattr(
        controller,
        "_probe_server",
        lambda _spec: True,
    )

    assert controller._start_servers(request) is True

    assert controller.text_endpoints_file.read_text(
        encoding="utf-8"
    ).strip() == controller.text_server.base_url
    assert controller.image_endpoints_file.read_text(
        encoding="utf-8"
    ).strip() == controller.image_server.base_url
    assert protocol.acknowledgement_matches(
        controller.paths.acknowledgement,
        request,
        status=protocol.SERVING_STATUS,
    )
    controller.processes.clear()


def test_completed_text_queue_uses_both_gpus_for_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "model-jobs.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE model_jobset_pairs (
                identity TEXT PRIMARY KEY,
                text_kind TEXT NOT NULL,
                image_kind TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE TABLE jobs (kind TEXT NOT NULL, status TEXT NOT NULL)"
        )
        connection.execute(
            """
            INSERT INTO model_jobset_pairs (
                identity, text_kind, image_kind, updated_at
            ) VALUES ('pair', 'text-kind', 'image-kind', 1)
            """
        )
        connection.executemany(
            "INSERT INTO jobs (kind, status) VALUES (?, ?)",
            [
                ("text-kind", "success"),
                ("image-kind", "success"),
                ("image-kind", "pending"),
                ("image-kind", "leased"),
            ],
        )
    controller = make_controller(
        tmp_path,
        extra_args=(
            "--model_jobs_database",
            str(database_path),
        ),
    )
    request = publish_request(
        controller,
        state=protocol.BORROWABLE_STATE,
    )
    processes = iter((FakeProcess(111), FakeProcess(112)))
    started_specs: list[borrower.VllmServerSpec] = []

    def start(spec, **_kwargs):
        started_specs.append(spec)
        return next(processes)

    monkeypatch.setattr(borrower, "start_server", start)
    monkeypatch.setattr(
        controller,
        "_probe_server",
        lambda _spec: True,
    )

    assert controller.desired_layout(force=True) == (
        borrower.IMAGE_ONLY_LAYOUT
    )
    assert controller._start_servers(request) is True

    assert [spec.gpu for spec in started_specs] == [
        controller.image_server.gpu,
        controller.secondary_image_server.gpu,
    ]
    assert controller.text_endpoints_file.read_text(
        encoding="utf-8"
    ) == ""
    assert controller.image_endpoints_file.read_text(
        encoding="utf-8"
    ).splitlines() == [
        controller.image_server.base_url,
        controller.secondary_image_server.base_url,
    ]
    controller.processes.clear()


def test_reclaim_withdraws_endpoints_and_stops_groups_before_acknowledging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = make_controller(tmp_path)
    request = publish_request(
        controller,
        state=protocol.PRIORITY_REQUESTED_STATE,
    )
    text_process = FakeProcess(201)
    image_process = FakeProcess(202)
    controller.processes = {
        "text": text_process,
        "image": image_process,
    }
    protocol.atomic_write_endpoints(
        controller.text_endpoints_file,
        [controller.text_server.base_url],
    )
    protocol.atomic_write_endpoints(
        controller.image_endpoints_file,
        [controller.image_server.base_url],
    )
    signals: list[tuple[int, int]] = []

    monkeypatch.setattr(
        borrower,
        "process_group_alive",
        lambda process: process.alive,
    )

    def kill_group(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        process = (
            text_process if pid == text_process.pid else image_process
        )
        process.alive = False

    monkeypatch.setattr(borrower.os, "killpg", kill_group)
    original_ack = protocol.write_acknowledgement
    ack_observations: list[tuple[str, str, bool]] = []

    def assert_safe_ack(path, current_request, **kwargs):
        ack_observations.append(
            (
                controller.text_endpoints_file.read_text(encoding="utf-8"),
                controller.image_endpoints_file.read_text(encoding="utf-8"),
                bool(controller.processes),
            )
        )
        original_ack(path, current_request, **kwargs)

    monkeypatch.setattr(protocol, "write_acknowledgement", assert_safe_ack)

    controller._acknowledge_release(request)

    assert ack_observations == [("", "", False)]
    assert signals == [
        (text_process.pid, signal.SIGTERM),
        (image_process.pid, signal.SIGTERM),
    ]
    assert protocol.acknowledgement_matches(
        controller.paths.acknowledgement,
        request,
        status=protocol.RELEASED_STATUS,
    )


def test_stop_servers_reaps_a_terminated_process_group_leader(
    tmp_path: Path,
) -> None:
    controller = make_controller(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
        start_new_session=True,
        text=True,
    )
    controller.processes = {"text": process}

    try:
        controller.stop_servers(request=None, state="stopping")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    assert controller.processes == {}
    assert process.returncode == -signal.SIGTERM


def test_partial_server_launch_is_cleaned_without_publishing_an_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = make_controller(tmp_path)
    request = publish_request(
        controller,
        state=protocol.BORROWABLE_STATE,
    )
    text_process = FakeProcess(301)
    starts = 0

    def start(*_args, **_kwargs):
        nonlocal starts
        starts += 1
        if starts == 1:
            return text_process
        raise OSError("injected image launch failure")

    monkeypatch.setattr(borrower, "start_server", start)
    monkeypatch.setattr(
        borrower,
        "process_group_alive",
        lambda process: process.alive,
    )

    def kill_group(pid: int, _signum: int) -> None:
        assert pid == text_process.pid
        text_process.alive = False

    monkeypatch.setattr(os, "killpg", kill_group)

    assert controller._start_servers(request) is False
    assert controller.processes == {}
    assert controller.text_endpoints_file.read_text(encoding="utf-8") == ""
    assert controller.image_endpoints_file.read_text(encoding="utf-8") == ""


def test_reclaim_never_acknowledges_when_endpoint_withdrawal_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = make_controller(tmp_path)
    request = publish_request(
        controller,
        state=protocol.PRIORITY_REQUESTED_STATE,
    )
    process = FakeProcess(401)
    controller.processes = {"text": process}
    monkeypatch.setattr(
        controller,
        "clear_endpoints",
        lambda: (_ for _ in ()).throw(OSError("injected fsync failure")),
    )
    monkeypatch.setattr(
        borrower,
        "process_group_alive",
        lambda current: current.alive,
    )

    def kill_group(pid: int, _signum: int) -> None:
        assert pid == process.pid
        process.alive = False

    monkeypatch.setattr(borrower.os, "killpg", kill_group)

    with pytest.raises(OSError, match="fsync failure"):
        controller._acknowledge_release(request)

    assert process.alive is False
    assert controller.processes == {}
    assert not protocol.acknowledgement_matches(
        controller.paths.acknowledgement,
        request,
        status=protocol.RELEASED_STATUS,
    )
