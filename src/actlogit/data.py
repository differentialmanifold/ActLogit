from __future__ import annotations

import json
from pathlib import Path

from actlogit.schema import TrainingRecord


def load_records(path: str | Path) -> list[TrainingRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(TrainingRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
