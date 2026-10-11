# Exported shared helpers; bodies unchanged. See CODE_PROVENANCE.json.
from __future__ import annotations
import csv,json,os
from pathlib import Path
from typing import Any,Iterable
import numpy as np
from PIL import Image
from DINOv3.relation_reliability.datasets import Record

def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def _load_mask(record: Record, output_shape: tuple[int, int]) -> np.ndarray:
    if record.label == 0:
        return np.zeros(output_shape, dtype=bool)
    if not record.mask_path:
        raise FileNotFoundError(f"MVTec anomaly mask is missing: {record.path}")
    with Image.open(record.mask_path) as source:
        mask = source.convert("L").resize(
            (output_shape[1], output_shape[0]), Image.Resampling.NEAREST,
        )
    return np.asarray(mask, dtype=np.uint8) > 0
