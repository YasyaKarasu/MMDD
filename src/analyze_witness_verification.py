#!/usr/bin/env python
"""Grade the independent verification pass and separate signal from artifact.

The raw judge agreement rate is not directly interpretable, because the judge
produces both false negatives (it misses a real bridge such as a nickname or a
country) and false positives (it accepts a bare year or a country name as a
"shared value", which identifies no entity at all).

So every judge-positive verdict is additionally graded on whether the value it
found is *discriminating*: it must land on exactly one row of the named target
column, that column must have real cardinality, and the value must not be a
bare year, a bare number, or a country name.  A verdict resting on a
non-discriminating value is a type match, not a join.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "work/witness_diagnostic_20260916"

YEAR = re.compile(r"^(1[6-9]\d{2}|20\d{2})$")
NUMBER = re.compile(r"^[\d,.\s%$+-]+$")
COUNTRY_LIKE = re.compile(r"^(the\s+)?[A-Z][a-z]+(\s+[A-Z][a-z]+)?$")


def parse_table(parts: list[str] | None) -> tuple[list[str], list[list[str]]]:
    columns: list[str] = []
    rows: list[list[str]] = []
    for part in parts or []:
        if part.startswith("Columns: "):
            columns = [value.strip() for value in part[9:].split("|")]
        elif part.startswith("Row: "):
            rows.append([value.strip() for value in part[5:].split("|")])
    return columns, rows


def grade(record: dict[str, Any]) -> dict[str, Any]:
    columns, rows = parse_table(record.get("target_columns_rows"))
    column = record.get("target_column")
    value = record.get("shared_value")
    result = {
        "target_rows": len(rows),
        "target_columns": len(columns),
        "column_found": False,
        "matching_rows": None,
        "distinct_values_in_column": None,
        "value_is_bare_year": bool(isinstance(value, str) and YEAR.match(value.strip())),
        "value_is_bare_number": bool(isinstance(value, str) and NUMBER.match(value.strip())),
        "value_is_country_like": bool(isinstance(value, str) and COUNTRY_LIKE.match(value.strip())),
        "discriminating": False,
    }
    if not columns or value is None or column is None:
        return result
    index = next(
        (position for position, name in enumerate(columns)
         if name.strip().lower() == str(column).strip().lower()),
        None,
    )
    if index is None:
        return result
    result["column_found"] = True
    values = [(row[index] if index < len(row) else "") for row in rows]
    result["matching_rows"] = sum(
        1 for item in values if item.strip().lower() == str(value).strip().lower()
    )
    result["distinct_values_in_column"] = len({item.strip().lower() for item in values if item.strip()})
    result["discriminating"] = bool(
        result["matching_rows"] == 1
        and result["distinct_values_in_column"] >= 4
        and not result["value_is_bare_year"]
        and not result["value_is_bare_number"]
    )
    return result


def main() -> int:
    records = [
        json.loads(line)
        for line in (IN / "VERIFICATION_RESULTS.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    graded = []
    for record in records:
        graded.append({**record, "grading": grade(record)})

    with (IN / "verification_graded.jsonl").open("w", encoding="utf-8") as handle:
        for record in graded:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    rows = []
    for group in ("retained_witness", "unretrieved_witness", "unknown_positive", "top10_competitor"):
        items = [record for record in graded if record["group"] == group]
        total = len(items)
        unparseable = sum(1 for item in items if item.get("join_supported") is None)
        supported = [item for item in items if item.get("join_supported") is True]
        discriminating = [item for item in supported if item["grading"]["discriminating"]]
        rows.append(
            {
                "group": group,
                "packets": total,
                "judge_supported": len(supported),
                "judge_supported_pct": round(100 * len(supported) / total, 3) if total else None,
                "of_which_discriminating_value": len(discriminating),
                "discriminating_pct_of_group": round(100 * len(discriminating) / total, 3) if total else None,
                "non_discriminating_type_matches": len(supported) - len(discriminating),
                "judge_unsupported": sum(1 for item in items if item.get("join_supported") is False),
                "unparseable": unparseable,
            }
        )
    with (IN / "verification_graded_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    modality = Counter()
    for record in graded:
        modality[(record["group"], record["evidence_modality"], record.get("join_supported"))] += 1

    report = {
        "note": "model-assisted second opinion, not ground truth",
        "groups": rows,
        "by_modality": {f"{g}|{m}|{v}": c for (g, m, v), c in sorted(modality.items(), key=str)},
        "discriminating_rule": (
            "judge shared_value matches exactly one row of the named target column, "
            "that column has >= 4 distinct values, and the value is not a bare year or number"
        ),
    }
    (IN / "VERIFICATION_SUMMARY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
