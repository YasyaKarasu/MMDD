"""Measure the zero-training PCA dimension ceiling for Stage-1 direct retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import evaluate_direct_retrieval
from mmdd_stage1.features import OBJECT_TYPES, FeatureStore
from mmdd_stage1.models import ProjectedIdentityStudentJoinabilityModel
from mmdd_stage1.pca import PCA_SPECTRUM_FORMAT_VERSION, compute_pca_spectrum
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    build_indices,
    checkpoint_fingerprint,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
)
from PIL import Image, ImageDraw, ImageFont


def _projection_fingerprint(projection: torch.Tensor, corpus_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"stage1-zero-training-pca-v1\0")
    digest.update(corpus_sha256.encode("ascii"))
    digest.update(str(tuple(projection.shape)).encode("ascii"))
    digest.update(projection.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_or_compute_spectrum(
    path: Path,
    embeddings: torch.Tensor,
    *,
    corpus_sha256: str,
    max_components: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": PCA_SPECTRUM_FORMAT_VERSION,
            "artifact_kind": "stage1_pca_spectrum",
            "input_dim": embeddings.shape[1],
            "objects": embeddings.shape[0],
            "corpus_sha256": corpus_sha256,
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"{path}: PCA spectrum does not match this corpus")
        projection = payload.get("projection")
        eigenvalues = payload.get("eigenvalues")
        explained = payload.get("explained_variance_ratio")
        if (
            not isinstance(projection, torch.Tensor)
            or projection.shape[0] < max_components
            or projection.shape[1] != embeddings.shape[1]
            or not isinstance(eigenvalues, torch.Tensor)
            or eigenvalues.shape != (embeddings.shape[1],)
            or not isinstance(explained, torch.Tensor)
            or explained.shape != (embeddings.shape[1],)
        ):
            raise ValueError(f"{path}: incomplete PCA spectrum artifact")
        return payload

    projection, mean, eigenvalues, explained = compute_pca_spectrum(
        embeddings,
        max_components,
        device=device,
        batch_size=batch_size,
    )
    payload = {
        "format_version": PCA_SPECTRUM_FORMAT_VERSION,
        "artifact_kind": "stage1_pca_spectrum",
        "input_dim": embeddings.shape[1],
        "max_components": max_components,
        "objects": embeddings.shape[0],
        "corpus_sha256": corpus_sha256,
        "mean": mean,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained,
        "projection": projection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def _load_or_build_dimension_index(
    model: ProjectedIdentityStudentJoinabilityModel,
    store: FeatureStore,
    table_ids: list[str],
    index_dir: Path,
    *,
    device: torch.device,
    projection_sha256: str,
    corpus_sha256: str,
    args: argparse.Namespace,
) -> StudentANNIndices:
    manifest_path = index_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "hnsw_m": args.hnsw_m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{manifest_path}: HNSW settings differ from this run")
    else:
        direct_ids = {object_type: [] for object_type in OBJECT_TYPES}
        direct_ids["table"] = table_ids
        build_indices(
            model,
            store,
            direct_ids,
            index_dir,
            device=device,
            checkpoint_sha256=projection_sha256,
            corpus_sha256=corpus_sha256,
            batch_size=args.batch_size,
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    return StudentANNIndices(
        model,
        store,
        index_dir,
        device=device,
        checkpoint_sha256=projection_sha256,
        corpus_sha256=corpus_sha256,
        destination_types=("table",),
    )


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _dashed_horizontal(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int],
    *,
    fill: str,
    width: int,
    dash: int,
) -> None:
    x0, x1, y = xy
    for start in range(x0, x1, dash * 2):
        draw.line((start, y, min(start + dash, x1), y), fill=fill, width=width)


def _write_chart(
    path: Path,
    explained: list[float],
    probes: list[dict[str, Any]],
    *,
    raw_recall: float,
    threshold_fraction: float,
    corpus_objects: int,
    dev_queries: int,
) -> None:
    scale = 2
    width, height = 1600 * scale, 1000 * scale
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    ink, quiet, grid = "#202124", "#5f6368", "#e1e5ea"
    blue, orange = "#2468a2", "#d97706"
    title_font = _font(34 * scale, bold=True)
    subtitle_font = _font(18 * scale)
    panel_font = _font(22 * scale, bold=True)
    axis_font = _font(16 * scale)
    label_font = _font(15 * scale, bold=True)

    draw.text(
        (90 * scale, 45 * scale),
        "Stage-1 zero-training PCA dimension ceiling",
        fill=ink,
        font=title_font,
    )
    subtitle = (
        f"P = U_d^T, R = I, no training | corpus n={corpus_objects:,} | "
        f"dev queries n={dev_queries:,} | HNSW inner product"
    )
    draw.text((90 * scale, 94 * scale), subtitle, fill=quiet, font=subtitle_font)

    panels = ((90, 170, 1510, 515), (90, 610, 1510, 925))
    x_min, x_max = 0.0, math.log2(len(explained))
    x_ticks = [
        value
        for value in (1, 16, 64, 128, 256, 512, 1024, 2048, 4096)
        if value <= len(explained)
    ]

    def x_position(dimension: int, panel: tuple[int, int, int, int]) -> int:
        left, _top, right, _bottom = panel
        ratio = (math.log2(dimension) - x_min) / (x_max - x_min)
        return int((left + ratio * (right - left)) * scale)

    top_panel = panels[0]
    left, top, right, bottom = (value * scale for value in top_panel)
    draw.text(
        (left, (top_panel[1] - 45) * scale),
        "Cumulative explained variance",
        fill=ink,
        font=panel_font,
    )
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(bottom - value * (bottom - top))
        draw.line((left, y, right, y), fill=grid, width=scale)
        draw.text(
            (30 * scale, y - 10 * scale), f"{value:.0%}", fill=quiet, font=axis_font
        )
    points = [
        (
            x_position(dimension, top_panel),
            int(bottom - value * (bottom - top)),
        )
        for dimension, value in enumerate(explained, 1)
    ]
    draw.line(points, fill=blue, width=3 * scale, joint="curve")

    bottom_panel = panels[1]
    left, top, right, bottom = (value * scale for value in bottom_panel)
    draw.text(
        (left, (bottom_panel[1] - 45) * scale),
        "Dev direct Recall@10",
        fill=ink,
        font=panel_font,
    )
    threshold = threshold_fraction * raw_recall
    max_probe = max(float(row["direct"]["recall@10"]) for row in probes)
    y_max = min(1.0, max(0.1, raw_recall * 1.2, max_probe * 1.2))
    for step in range(5):
        value = y_max * step / 4
        y = int(bottom - value / y_max * (bottom - top))
        draw.line((left, y, right, y), fill=grid, width=scale)
        draw.text(
            (30 * scale, y - 10 * scale), f"{value:.1%}", fill=quiet, font=axis_font
        )
    raw_y = int(bottom - raw_recall / y_max * (bottom - top))
    threshold_y = int(bottom - threshold / y_max * (bottom - top))
    _dashed_horizontal(
        draw, (left, right, raw_y), fill=ink, width=2 * scale, dash=9 * scale
    )
    _dashed_horizontal(
        draw, (left, right, threshold_y), fill=quiet, width=2 * scale, dash=3 * scale
    )
    draw.text(
        (left + 12 * scale, raw_y - 28 * scale),
        f"raw {raw_recall:.1%}",
        fill=ink,
        font=axis_font,
    )
    draw.text(
        (left + 12 * scale, threshold_y + 8 * scale),
        f"90% of raw {threshold:.1%}",
        fill=quiet,
        font=axis_font,
    )
    probe_points = []
    for row in probes:
        value = float(row["direct"]["recall@10"])
        point = (
            x_position(int(row["dimension"]), bottom_panel),
            int(bottom - value / y_max * (bottom - top)),
        )
        probe_points.append(point)
    draw.line(probe_points, fill=orange, width=4 * scale)
    radius = 6 * scale
    for point, row in zip(probe_points, probes):
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill="#ffffff",
            outline=orange,
            width=3 * scale,
        )
        value = float(row["direct"]["recall@10"])
        text = f"{int(row['dimension'])}: {value:.1%}"
        draw.text(
            (point[0] - 42 * scale, point[1] - 32 * scale),
            text,
            fill=orange,
            font=label_font,
        )

    for panel in panels:
        left, top, right, bottom = (value * scale for value in panel)
        draw.line((left, top, left, bottom), fill=ink, width=2 * scale)
        draw.line((left, bottom, right, bottom), fill=ink, width=2 * scale)
        for value in x_ticks:
            x = x_position(value, panel)
            draw.line((x, bottom, x, bottom + 7 * scale), fill=ink, width=2 * scale)
            label = str(value)
            label_width = draw.textlength(label, font=axis_font)
            draw.text(
                (x - label_width / 2, bottom + 10 * scale),
                label,
                fill=quiet,
                font=axis_font,
            )
    draw.text(
        (690 * scale, 958 * scale),
        "Embedding dimension (log2 scale)",
        fill=quiet,
        font=axis_font,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width // scale, height // scale), Image.Resampling.LANCZOS).save(path)


def _write_tables(
    output_dir: Path,
    explained: list[float],
    probes: list[dict[str, Any]],
) -> None:
    with (output_dir / "variance_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("dimension", "explained_variance_ratio")
        )
        writer.writeheader()
        writer.writerows(
            {"dimension": dimension, "explained_variance_ratio": value}
            for dimension, value in enumerate(explained, 1)
        )
    with (output_dir / "dimension_probe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "dimension",
            "explained_variance_ratio",
            "direct_recall_at_10",
            "fraction_of_raw_direct_recall_at_10",
            "meets_raw_fraction_threshold",
            "index_bytes",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in probes:
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "explained_variance_ratio": row["explained_variance_ratio"],
                    "direct_recall_at_10": row["direct"]["recall@10"],
                    "fraction_of_raw_direct_recall_at_10": row[
                        "fraction_of_raw_direct_recall_at_10"
                    ],
                    "meets_raw_fraction_threshold": row["meets_raw_fraction_threshold"],
                    "index_bytes": row["index_bytes"],
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dimensions = sorted(set(args.dimensions))
    if not dimensions or dimensions[0] <= 0:
        raise ValueError("--dimensions must contain positive integers")
    if not 0 < args.raw_fraction_threshold <= 1:
        raise ValueError("--raw-fraction-threshold must be in (0, 1]")
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    ids_by_type = load_corpus_ids(corpus_path, store)
    object_ids = [
        object_id
        for object_type in OBJECT_TYPES
        for object_id in ids_by_type[object_type]
    ]
    embeddings = store.preload_embedding_matrix(object_ids)
    if dimensions[-1] > embeddings.shape[1]:
        raise ValueError("A requested PCA dimension exceeds the embedding dimension")

    spectrum_path = output_dir / "pca_spectrum.pt"
    spectrum = _load_or_compute_spectrum(
        spectrum_path,
        embeddings,
        corpus_sha256=corpus_sha256,
        max_components=dimensions[-1],
        device=device,
        batch_size=args.covariance_batch_size,
    )
    projection = spectrum["projection"].float()
    explained_tensor = spectrum["explained_variance_ratio"].float()
    explained = [float(value) for value in explained_tensor]

    examples = [
        example
        for path in args.dev_data
        for example in load_target_examples(Path(path), split="dev")
    ]
    if args.raw_index:
        raw_indices = RawEmbeddingANNIndices(
            store,
            Path(args.raw_index),
            corpus_sha256=corpus_sha256,
            destination_types=("table",),
        )
    else:
        direct_ids = {object_type: [] for object_type in OBJECT_TYPES}
        direct_ids["table"] = ids_by_type["table"]
        raw_indices = load_or_build_raw_embedding_indices(
            store,
            direct_ids,
            output_dir / "raw_embedding_index",
            corpus_sha256=corpus_sha256,
            batch_size=args.batch_size,
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    raw_direct = evaluate_direct_retrieval(
        examples, raw_indices, direct_k=args.direct_k
    )
    raw_recall = float(raw_direct["recall@10"])
    if raw_recall <= 0:
        raise ValueError(
            "Raw direct Recall@10 must be positive to define a ceiling threshold"
        )
    del raw_indices

    probes = []
    for dimension in dimensions:
        basis = projection[:dimension]
        projection_sha256 = _projection_fingerprint(basis, corpus_sha256)
        model = ProjectedIdentityStudentJoinabilityModel(basis).to(device)
        index_dir = output_dir / "indices" / f"pca_{dimension}"
        indices = _load_or_build_dimension_index(
            model,
            store,
            ids_by_type["table"],
            index_dir,
            device=device,
            projection_sha256=projection_sha256,
            corpus_sha256=corpus_sha256,
            args=args,
        )
        direct = evaluate_direct_retrieval(examples, indices, direct_k=args.direct_k)
        fraction = float(direct["recall@10"]) / raw_recall
        index_bytes = sum(path.stat().st_size for path in index_dir.glob("*.hnsw"))
        row = {
            "dimension": dimension,
            "explained_variance_ratio": explained[dimension - 1],
            "projection_sha256": projection_sha256,
            "index_dir": str(index_dir),
            "index_bytes": index_bytes,
            "direct": direct,
            "fraction_of_raw_direct_recall_at_10": fraction,
            "meets_raw_fraction_threshold": fraction >= args.raw_fraction_threshold,
        }
        probes.append(row)
        (index_dir / "metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        del indices, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_dimension = next(
        (row["dimension"] for row in probes if row["meets_raw_fraction_threshold"]),
        None,
    )
    result = {
        "probe": "zero_training_pca_dimension_ceiling",
        "training_steps": 0,
        "projection": "P = U_d^T from centered corpus covariance; embeddings are not centered at retrieval time",
        "relation": "R = I",
        "features": str(Path(args.features)),
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "corpus_objects": len(object_ids),
        "indexed_table_objects": len(ids_by_type["table"]),
        "embedding_dim": embeddings.shape[1],
        "dev_data": [str(Path(path)) for path in args.dev_data],
        "raw_direct": raw_direct,
        "raw_fraction_threshold": args.raw_fraction_threshold,
        "raw_direct_recall_at_10_threshold": args.raw_fraction_threshold * raw_recall,
        "selected_dimension": selected_dimension,
        "dimensions": probes,
        "artifacts": {
            "pca_spectrum": str(spectrum_path),
            "variance_curve_csv": str(output_dir / "variance_curve.csv"),
            "dimension_probe_csv": str(output_dir / "dimension_probe.csv"),
            "figure": str(output_dir / "pca_dimension_ceiling.png"),
        },
    }
    _write_tables(output_dir, explained, probes)
    _write_chart(
        output_dir / "pca_dimension_ceiling.png",
        explained,
        probes,
        raw_recall=raw_recall,
        threshold_fraction=args.raw_fraction_threshold,
        corpus_objects=len(object_ids),
        dev_queries=int(raw_direct["queries"]),
    )
    result_path = output_dir / "summary.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dev-data", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-index")
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=(128, 256, 512, 1024, 2048)
    )
    parser.add_argument("--raw-fraction-threshold", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--covariance-batch-size", type=int, default=4096)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--direct-k", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
