#!/usr/bin/env python
"""Run the staged WDC 200K builder with OpenAI model inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import build_wdc200k_mm_joinability_dataset as pipeline
from openai_attribute_extractor import (
    OpenAIAttributeExtractor,
    current_process_usage,
    load_openai_environment_file,
    openai_inference_identity,
    summarize_usage_journal,
    validate_api_base_url,
)
from stage1_io import stable_hash, write_json


DEFAULT_OPENAI_MODEL = "gpt-5.6"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_ENV_FILE = Path(".env.openai")
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class OpenAIRun:
    config: pipeline.PipelineConfig
    extractor: OpenAIAttributeExtractor
    inference_identity: dict[str, Any]
    inference_fingerprint: str
    usage_journal_path: Path


def add_openai_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("OpenAI Chat Completions")
    group.add_argument(
        "--openai_model",
        default=DEFAULT_OPENAI_MODEL,
        help=(
            "OpenAI model ID used for both text and image extraction "
            f"(default: {DEFAULT_OPENAI_MODEL})."
        ),
    )
    group.add_argument(
        "--openai_base_url",
        "--openai_api_base_url",
        dest="openai_base_url",
        default=os.environ.get(
            "OPENAI_BASE_URL",
            DEFAULT_OPENAI_BASE_URL,
        ),
        help=(
            "Chat Completions base URL ending in /v1. Defaults to "
            "$OPENAI_BASE_URL when set, otherwise "
            f"{DEFAULT_OPENAI_BASE_URL}. --openai_api_base_url is a "
            "backward-compatible alias."
        ),
    )
    group.add_argument(
        "--openai_api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; the key is never persisted.",
    )
    group.add_argument(
        "--openai_env_file",
        default="",
        help=(
            "Chmod-600 dotenv file containing OPENAI_API_KEY and/or "
            "OPENAI_BASE_URL. Defaults to ./.env.openai when that file "
            "exists. Values override the inherited environment and are "
            "never persisted."
        ),
    )
    group.add_argument(
        "--openai_work_root",
        default="",
        help=(
            "Root for provider-isolated inference runs. The inference fingerprint "
            "is appended automatically. If omitted, --work_dir is treated as this "
            "root, then <output parent>/work_wdc_200k_openai is used."
        ),
    )
    group.add_argument(
        "--openai_reasoning_effort",
        choices=("omit", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
        help=(
            "Chat Completions reasoning effort. 'none' preserves the existing builder's "
            "disabled-thinking behavior and minimizes reasoning tokens."
        ),
    )
    group.add_argument(
        "--openai_verbosity",
        choices=("low", "medium", "high"),
        default="low",
    )
    group.add_argument(
        "--openai_max_output_tokens",
        type=int,
        default=1024,
    )
    group.add_argument(
        "--openai_image_detail",
        choices=("auto", "low", "high"),
        default="auto",
    )
    group.add_argument(
        "--openai_image_max_pixels",
        type=int,
        default=512_000,
        help="Resize cached local images to this pixel budget; 0 sends originals.",
    )
    group.add_argument(
        "--openai_context_retry_image_max_pixels",
        type=int,
        default=262_144,
    )
    group.add_argument(
        "--openai_max_inflight",
        type=int,
        default=4,
        help="Shared maximum in-flight requests across text and image workers.",
    )
    group.add_argument(
        "--openai_requests_per_minute",
        type=int,
        default=0,
        help="Optional shared request-rate limit; 0 disables proactive RPM pacing.",
    )
    group.add_argument(
        "--openai_tokens_per_minute",
        type=int,
        default=0,
        help="Optional shared approximate token-rate limit; 0 disables TPM pacing.",
    )
    group.add_argument(
        "--openai_retry_max_seconds",
        type=float,
        default=60.0,
        help="Maximum delay for exponential or server-requested retry backoff.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    environment_parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )
    environment_parser.add_argument("--openai_env_file", default="")
    environment_args, _unknown = environment_parser.parse_known_args(
        arguments
    )
    environment_path = (
        Path(environment_args.openai_env_file)
        if environment_args.openai_env_file
        else DEFAULT_OPENAI_ENV_FILE
    )
    if environment_args.openai_env_file or environment_path.exists():
        load_openai_environment_file(environment_path)
    return pipeline.parse_args(
        arguments,
        configure_parser=add_openai_arguments,
    )


def _safe_model_component(model: str) -> str:
    value = _SAFE_PATH_COMPONENT.sub("-", model.strip()).strip("-._")
    return (value or "model")[:64]


def prepare_openai_run(args: argparse.Namespace) -> OpenAIRun:
    model = str(args.openai_model or "").strip()
    api_base_url = validate_api_base_url(args.openai_base_url)
    key_environment = str(args.openai_api_key_env or "").strip()
    if not model:
        raise ValueError("--openai_model must not be empty")
    if not api_base_url:
        raise ValueError("--openai_base_url must not be empty")
    if not key_environment or "=" in key_environment or "\x00" in key_environment:
        raise ValueError("--openai_api_key_env is not a valid environment variable name")
    if args.openai_max_output_tokens <= 0:
        raise ValueError("--openai_max_output_tokens must be positive")
    if args.openai_image_max_pixels < 0:
        raise ValueError("--openai_image_max_pixels must be non-negative")
    if args.openai_context_retry_image_max_pixels <= 0:
        raise ValueError(
            "--openai_context_retry_image_max_pixels must be positive"
        )
    if args.openai_max_inflight <= 0:
        raise ValueError("--openai_max_inflight must be positive")
    if min(
        args.openai_requests_per_minute,
        args.openai_tokens_per_minute,
    ) < 0:
        raise ValueError("OpenAI RPM and TPM limits must be non-negative")
    if args.openai_retry_max_seconds < 0:
        raise ValueError("--openai_retry_max_seconds must be non-negative")
    if args.openai_work_root and args.work_dir:
        raise ValueError("use only one of --openai_work_root and --work_dir")

    identity = openai_inference_identity(
        model=model,
        api_base_url=api_base_url,
        reasoning_effort=args.openai_reasoning_effort,
        verbosity=args.openai_verbosity,
        max_output_tokens=args.openai_max_output_tokens,
        image_detail=args.openai_image_detail,
        image_max_pixels=args.openai_image_max_pixels,
        context_retry_image_max_pixels=(
            args.openai_context_retry_image_max_pixels
        ),
    )
    fingerprint = stable_hash(
        "wdc200k-openai-inference-v1",
        json.dumps(identity, ensure_ascii=False, sort_keys=True),
        length=24,
    )
    output_dir = Path(args.output_dir).resolve()
    if args.openai_work_root:
        work_root = Path(args.openai_work_root).resolve()
    elif args.work_dir:
        work_root = Path(args.work_dir).resolve()
    else:
        work_root = output_dir.parent / "work_wdc_200k_openai"
    args.work_dir = str(
        work_root
        / "openai_model_runs"
        / f"{_safe_model_component(model)}-{fingerprint}"
    )

    provider_model_identity = f"openai-chat:{model}:{fingerprint}"
    args.text_model_name = provider_model_identity
    args.image_model_name = provider_model_identity
    args.text_model_base_url = api_base_url
    args.image_model_base_url = api_base_url
    args.text_model_base_urls = None
    args.image_model_base_urls = None
    args.text_model_base_urls_file = None
    args.image_model_base_urls_file = None
    args.text_model_api_key = None
    args.image_model_api_key = None

    config = pipeline.PipelineConfig.from_args(args)
    usage_journal_path = config.work_dir / "openai_usage.jsonl"
    extractor = OpenAIAttributeExtractor(
        api_key=os.environ.get(key_environment),
        model=model,
        api_base_url=api_base_url,
        timeout_seconds=args.model_timeout_seconds,
        max_retries=args.model_max_retries,
        retry_sleep_seconds=args.model_retry_sleep_seconds,
        max_output_tokens=args.openai_max_output_tokens,
        reasoning_effort=args.openai_reasoning_effort,
        verbosity=args.openai_verbosity,
        image_detail=args.openai_image_detail,
        image_max_pixels=args.openai_image_max_pixels,
        context_retry_image_max_pixels=(
            args.openai_context_retry_image_max_pixels
        ),
        max_inflight=args.openai_max_inflight,
        requests_per_minute=args.openai_requests_per_minute,
        tokens_per_minute=args.openai_tokens_per_minute,
        retry_max_seconds=args.openai_retry_max_seconds,
        usage_journal_path=usage_journal_path,
    )
    write_json(
        config.work_dir / "openai_run_config.json",
        {
            "stage": "wdc200k_openai_run",
            "schema_version": "wdc200k-openai-run-v1",
            "inference_fingerprint": fingerprint,
            "inference_identity": identity,
            "download_cache_dir": str(config.cache_dir),
            "inference_work_dir": str(config.work_dir),
            "usage_journal": str(usage_journal_path),
            "request_controls": extractor.request_controller.summary(),
            "retry_max_seconds": extractor.retry_max_seconds,
        },
    )
    return OpenAIRun(
        config=config,
        extractor=extractor,
        inference_identity=identity,
        inference_fingerprint=fingerprint,
        usage_journal_path=usage_journal_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    openai_run: OpenAIRun | None = None
    try:
        openai_run = prepare_openai_run(parse_args(argv))
        result = pipeline.run_pipeline(
            openai_run.config,
            extractor=openai_run.extractor,
        )
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2

    print(
        json.dumps(
            {
                "status": result.status,
                "stage": result.stage,
                "statistics_archives": result.statistics_archives,
                "counters": result.counters,
                "output_manifest": (
                    None
                    if result.output_manifest is None
                    else str(result.output_manifest)
                ),
                "openai": {
                    "model": openai_run.inference_identity["model"],
                    "inference_fingerprint": (
                        openai_run.inference_fingerprint
                    ),
                    "download_cache_dir": str(openai_run.config.cache_dir),
                    "inference_work_dir": str(openai_run.config.work_dir),
                    "current_process_usage": current_process_usage(
                        openai_run.extractor
                    ),
                    "cumulative_usage": summarize_usage_journal(
                        openai_run.usage_journal_path
                    ),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
