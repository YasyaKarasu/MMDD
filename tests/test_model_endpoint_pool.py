import json
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

from model_endpoint_pool import (  # noqa: E402
    ModelEndpointScheduler,
    load_model_endpoint_config,
)


def _write_config(tmp_path: Path, endpoints: list[dict]) -> Path:
    path = tmp_path / "model-endpoints.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "mmdd-model-endpoints-v1",
                "served_model_name": "Qwen3.5-9B",
                "endpoints": endpoints,
            }
        ),
        encoding="utf-8",
    )
    return path


def _endpoint(
    endpoint_id: str,
    base_url: str,
    *,
    pool: str = "remote",
    text: int,
    image: int,
    total: int,
) -> dict:
    return {
        "endpoint_id": endpoint_id,
        "base_url": base_url,
        "pool": pool,
        "max_inflight": {
            "text": text,
            "image": image,
            "total": total,
        },
    }


def test_config_represents_one_multimodal_endpoint_once(tmp_path):
    config = load_model_endpoint_config(
        _write_config(
            tmp_path,
            [
                _endpoint(
                    "a100-0",
                    "http://127.0.0.1:18011/v1/",
                    text=32,
                    image=8,
                    total=36,
                )
            ],
        )
    )

    assert config.served_model_name == "Qwen3.5-9B"
    assert config.endpoints[0].base_url == "http://127.0.0.1:18011/v1"
    assert config.endpoints[0].supports("text")
    assert config.endpoints[0].supports("image")


def test_duplicate_physical_url_is_rejected(tmp_path):
    path = _write_config(
        tmp_path,
        [
            _endpoint(
                "remote-text",
                "http://127.0.0.1:18011/v1",
                text=4,
                image=0,
                total=4,
            ),
            _endpoint(
                "remote-image",
                "http://127.0.0.1:18011/v1",
                text=0,
                image=2,
                total=2,
            ),
        ],
    )

    with pytest.raises(ValueError, match="physical model endpoint URL"):
        load_model_endpoint_config(path)


def test_shared_total_limit_blocks_other_modality(tmp_path):
    scheduler = ModelEndpointScheduler(
        load_model_endpoint_config(
            _write_config(
                tmp_path,
                [
                    _endpoint(
                        "a100-0",
                        "http://127.0.0.1:18011/v1",
                        text=1,
                        image=1,
                        total=1,
                    )
                ],
            )
        )
    )
    text_endpoint = scheduler.acquire("remote", "text")
    image_acquired = threading.Event()
    image_released = threading.Event()

    def acquire_image() -> None:
        endpoint = scheduler.acquire("remote", "image")
        image_acquired.set()
        scheduler.release(endpoint, "image")
        image_released.set()

    thread = threading.Thread(target=acquire_image)
    thread.start()
    assert not image_acquired.wait(timeout=0.1)

    scheduler.release(text_endpoint, "text")

    assert image_acquired.wait(timeout=1)
    assert image_released.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_scheduler_balances_across_two_remote_replicas(tmp_path):
    scheduler = ModelEndpointScheduler(
        load_model_endpoint_config(
            _write_config(
                tmp_path,
                [
                    _endpoint(
                        "a100-0",
                        "http://127.0.0.1:18011/v1",
                        text=2,
                        image=1,
                        total=2,
                    ),
                    _endpoint(
                        "a100-1",
                        "http://127.0.0.1:18012/v1",
                        text=2,
                        image=1,
                        total=2,
                    ),
                ],
            )
        )
    )

    first = scheduler.acquire("remote", "text")
    second = scheduler.acquire("remote", "text")
    try:
        assert first.endpoint_id != second.endpoint_id
    finally:
        scheduler.release(first, "text")
        scheduler.release(second, "text")
