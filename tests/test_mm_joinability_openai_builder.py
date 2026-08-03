from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mm_joinability_dataset as builder  # noqa: E402
from build_mm_joinability_dataset_openai import (  # noqa: E402
    parse_args,
    prepare_openai_run,
)


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--input_dir",
        str(tmp_path / "input"),
        "--output_dir",
        str(tmp_path / "output"),
        "--cache_dir",
        str(tmp_path / "shared_cache"),
        "--openai_model",
        "gpt-test",
    ]


def _write_openai_env(path: Path) -> None:
    path.write_text(
        "OPENAI_API_KEY=secret-from-file\n"
        "OPENAI_BASE_URL=https://file-gateway.example.test/v1\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


@pytest.fixture(autouse=True)
def _isolate_default_openai_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)


def test_entitables_openai_run_is_fingerprinted_and_keeps_key_out_of_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    openai_run = prepare_openai_run(parse_args(_argv(tmp_path)))

    assert openai_run.args.text_model_name.startswith(
        "openai-chat:gpt-test:"
    )
    assert openai_run.args.text_model_name == openai_run.args.image_model_name
    assert openai_run.args.text_model_api_key is None
    assert openai_run.args.image_model_api_key is None
    assert openai_run.args.image_request_max_pixels == 512_000
    assert "openai_model_runs" in openai_run.inference_work_dir.parts
    assert openai_run.extractor.api_key == "secret-test-key"

    config = json.loads(
        (openai_run.inference_work_dir / "openai_run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["shared_cache_dir"] == str(
        (tmp_path / "shared_cache").resolve()
    )
    assert config["inference_work_dir"] == str(
        openai_run.inference_work_dir
    )
    assert "secret-test-key" not in json.dumps(config)
    assert "api_key" not in json.dumps(config)


def test_entitables_openai_env_file_overrides_inherited_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_file = tmp_path / ".env.openai"
    _write_openai_env(environment_file)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-shell-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://stale-gateway.example.test/v1",
    )

    openai_run = prepare_openai_run(
        parse_args(
            [
                *_argv(tmp_path),
                "--openai_env_file",
                str(environment_file),
            ]
        )
    )

    assert openai_run.args.openai_base_url == (
        "https://file-gateway.example.test/v1"
    )
    assert openai_run.extractor.api_key == "secret-from-file"
    config_text = (
        openai_run.inference_work_dir / "openai_run_config.json"
    ).read_text(encoding="utf-8")
    assert "secret-from-file" not in config_text
    assert "stale-shell-key" not in config_text


def test_entitables_default_openai_env_file_is_loaded_and_cli_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_openai_env(tmp_path / ".env.openai")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-shell-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://stale-gateway.example.test/v1",
    )

    from_file = prepare_openai_run(parse_args(_argv(tmp_path)))
    from_cli = prepare_openai_run(
        parse_args(
            [
                *_argv(tmp_path),
                "--openai_base_url",
                "https://override.example.test/v1",
            ]
        )
    )

    assert from_file.args.openai_base_url == (
        "https://file-gateway.example.test/v1"
    )
    assert from_file.extractor.api_key == "secret-from-file"
    assert from_cli.args.openai_base_url == (
        "https://override.example.test/v1"
    )


def test_entitables_openai_env_file_requires_private_permissions(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / ".env.openai"
    _write_openai_env(environment_file)
    environment_file.chmod(0o644)

    with pytest.raises(ValueError, match="chmod 600"):
        parse_args(
            [
                *_argv(tmp_path),
                "--openai_env_file",
                str(environment_file),
            ]
        )


def test_entitables_openai_base_url_uses_environment_and_cli_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.test/v1")

    from_environment = parse_args(_argv(tmp_path))
    from_cli = parse_args(
        [
            *_argv(tmp_path),
            "--openai_base_url",
            "https://override.example.test/v1",
        ]
    )

    assert (
        from_environment.openai_base_url
        == "https://gateway.example.test/v1"
    )
    assert from_cli.openai_base_url == "https://override.example.test/v1"


def test_entitables_legacy_openai_api_base_url_alias_is_preserved(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            *_argv(tmp_path),
            "--openai_api_base_url",
            "https://legacy.example.test/v1",
        ]
    )

    assert args.openai_base_url == "https://legacy.example.test/v1"


def test_entitables_openai_identity_controls_cache_and_work_isolation(
    tmp_path: Path,
) -> None:
    first = prepare_openai_run(parse_args(_argv(tmp_path)))
    resumed = prepare_openai_run(parse_args(_argv(tmp_path)))
    changed = prepare_openai_run(
        parse_args([*_argv(tmp_path), "--openai_reasoning_effort", "low"])
    )

    assert first.inference_fingerprint == resumed.inference_fingerprint
    assert first.inference_work_dir == resumed.inference_work_dir
    assert first.args.text_model_name == resumed.args.text_model_name
    assert changed.inference_work_dir != first.inference_work_dir
    assert changed.args.text_model_name != first.args.text_model_name
    assert Path(first.args.cache_dir).resolve() == Path(
        changed.args.cache_dir
    ).resolve()


def test_entitables_openai_runtime_limits_do_not_invalidate_model_cache(
    tmp_path: Path,
) -> None:
    first = prepare_openai_run(parse_args(_argv(tmp_path)))
    limited = prepare_openai_run(
        parse_args(
            [
                *_argv(tmp_path),
                "--openai_max_inflight",
                "2",
                "--openai_requests_per_minute",
                "60",
                "--openai_tokens_per_minute",
                "120000",
            ]
        )
    )

    assert limited.inference_fingerprint == first.inference_fingerprint
    assert limited.inference_work_dir == first.inference_work_dir
    assert limited.extractor.request_controller.summary() == {
        "max_inflight": 2,
        "requests_per_minute": 60,
        "tokens_per_minute": 120000,
    }


def test_entitables_openai_transient_error_aborts_resumable_batch(
    tmp_path: Path,
) -> None:
    class TransientExtractor:
        abort_on_transient_error = True

        @staticmethod
        def extract(*_args: object) -> dict[str, object]:
            raise builder.TransientModelEndpointError("HTTP 429")

    task = builder.ExtractionTask(
        order=0,
        cache_key="transient-call",
        source_table_id="source-1",
        source_row_id=0,
        entity_column_index=0,
        entity_column_name="Name",
        entity={
            "entity_id": "entity-1",
            "cell_text": "Alpha",
            "wiki_title": "Alpha",
            "row_attributes": [],
        },
        asset={
            "asset_id": "text-1",
            "asset_type": "text",
            "content": "Alpha",
        },
        candidate_attribute_names=["State"],
    )
    cache = builder.ExtractionCache(tmp_path / "cache.jsonl")
    args = SimpleNamespace(
        cache_failed_model_outputs=False,
        reparse_cached_model_outputs=True,
        refresh_invalid_model_cache=False,
        model_attribute_errors_path="",
    )

    with pytest.raises(
        builder.TransientModelEndpointError,
        match="run can be resumed",
    ):
        builder.resolve_extraction_tasks(
            extractor=TransientExtractor(),
            cache=cache,
            tasks=[task],
            args=args,
            state=builder.ModelConcurrencyState(1, 1),
        )

    assert cache.get("transient-call") is None


def test_entitables_builder_uses_injected_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    injected_extractor = object()

    def fake_selection(**kwargs: object) -> builder.ReplacementSelection:
        callback = kwargs["on_initial_batch_prepared"]
        assert callable(callback)
        callback([])
        return builder.ReplacementSelection([], [], 0, True, 0)

    def reject_local_extractor(_args: object) -> None:
        raise AssertionError("local extractor must not be constructed")

    monkeypatch.setattr(builder, "run_replacement_rounds", fake_selection)
    monkeypatch.setattr(builder, "LocalAttributeExtractor", reject_local_extractor)
    monkeypatch.setattr(
        builder,
        "write_sharded_jsonl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stop after extractor initialization")
        ),
    )
    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(tmp_path / "output"),
            "--max_source_tables",
            "0",
            "--no_wikipedia",
            "--no_model_progress",
        ]
    )

    with pytest.raises(RuntimeError, match="stop after extractor initialization"):
        builder.build_dataset(args, extractor=injected_extractor)
