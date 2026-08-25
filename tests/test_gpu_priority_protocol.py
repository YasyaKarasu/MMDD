import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

import gpu_priority_protocol as protocol


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def wall_time(self) -> float:
        return self.value

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def make_owner(
    tmp_path: Path,
    clock: FakeClock,
    *,
    sleep=None,
) -> protocol.PriorityGpuOwner:
    return protocol.PriorityGpuOwner(
        tmp_path,
        gpu_ids=("0", "1"),
        reclaim_timeout_seconds=2.0,
        borrower_stale_seconds=1.0,
        unregistered_grace_seconds=0.2,
        poll_seconds=0.1,
        wall_time=clock.wall_time,
        monotonic=clock.monotonic,
        sleep=sleep or clock.sleep,
    )


def test_owner_reclaims_without_waiting_for_an_unregistered_borrower(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    owner = make_owner(tmp_path, clock)

    owner.release_gpus(reason="network")
    request = owner.request_gpus(reason="model")

    assert request.state == protocol.PRIORITY_REQUESTED_STATE
    assert request.sequence == 2
    assert 100.2 <= clock.value <= 100.3
    assert protocol.read_priority_request(owner.paths.request) == request


def test_owner_ignores_stale_ack_and_waits_for_matching_release(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    callback_calls = 0
    owner: protocol.PriorityGpuOwner

    def borrower_step(seconds: float) -> None:
        nonlocal callback_calls
        callback_calls += 1
        request = protocol.read_priority_request(owner.paths.request)
        assert request is not None
        if request.state == protocol.PRIORITY_REQUESTED_STATE:
            protocol.write_acknowledgement(
                owner.paths.acknowledgement,
                request,
                borrower_id="borrower-1",
                status=protocol.RELEASED_STATUS,
                timestamp=clock.wall_time(),
            )
        clock.sleep(seconds)

    owner = make_owner(tmp_path, clock, sleep=borrower_step)
    old_request = owner.release_gpus(reason="network")
    protocol.write_acknowledgement(
        owner.paths.acknowledgement,
        old_request,
        borrower_id="borrower-1",
        status=protocol.RELEASED_STATUS,
        timestamp=clock.wall_time(),
    )
    protocol.write_borrower_status(
        owner.paths.borrower,
        borrower_id="borrower-1",
        state="serving",
        request=old_request,
        timestamp=clock.wall_time(),
    )

    request = owner.request_gpus(reason="model")

    assert callback_calls == 1
    assert protocol.acknowledgement_matches(
        owner.paths.acknowledgement,
        request,
        status=protocol.RELEASED_STATUS,
    )


def test_owner_refuses_reclaim_with_stale_borrower_heartbeat(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    owner = make_owner(tmp_path, clock)
    borrowable = owner.release_gpus(reason="network")
    protocol.write_borrower_status(
        owner.paths.borrower,
        borrower_id="crashed-borrower",
        state="serving",
        request=borrowable,
        timestamp=clock.wall_time() - 5.0,
    )

    with pytest.raises(RuntimeError, match="heartbeat is stale"):
        owner.request_gpus(reason="model")

    request = protocol.read_priority_request(owner.paths.request)
    assert request is not None
    assert request.state == protocol.PRIORITY_REQUESTED_STATE


def test_endpoint_snapshot_is_atomic_deduplicated_and_can_be_withdrawn(
    tmp_path: Path,
) -> None:
    path = tmp_path / "endpoints.txt"

    protocol.atomic_write_endpoints(
        path,
        [
            "http://127.0.0.1:18100/v1/",
            "http://127.0.0.1:18100/v1",
            " http://127.0.0.1:18101/v1 ",
        ],
    )
    assert path.read_text(encoding="utf-8").splitlines() == [
        "http://127.0.0.1:18100/v1",
        "http://127.0.0.1:18101/v1",
    ]

    protocol.atomic_write_endpoints(path, [])
    assert path.read_text(encoding="utf-8") == ""
