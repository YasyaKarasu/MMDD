from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plot_stage1_relation_drift import load_points, render_chart


def test_relation_drift_plot_loads_history_and_renders_png(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "epochs": [
                    {
                        "epoch": epoch,
                        "relation_drift": {"table_to_table": drift},
                        "dev_retrieval": {
                            "by_dataset": {
                                "entitables20k_v4": {
                                    "direct": {"recall@10": entitables}
                                },
                                "wdc2k_v2": {"direct": {"recall@10": wdc}},
                            }
                        },
                    }
                    for epoch, drift, entitables, wdc in (
                        (0, 0.0, 0.30, 0.60),
                        (1, 0.5, 0.32, 0.58),
                        (2, 1.0, 0.34, 0.55),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "relation_drift.png"

    points = load_points(history)
    render_chart(points, output)

    assert points[0]["epoch"] == 0
    assert points[-1]["wdc"] == 0.55
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (1400, 720)
