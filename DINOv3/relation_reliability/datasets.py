# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Sequence
from .protocol import MVTEC_AD_CATEGORIES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Record:
    path: str
    label: int
    mask_path: str | None
    category: str
    dataset: str
    domain: str
    shift: str
    role: str
    mask_capability: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _mvtec_root(root: Path) -> Path:
    for candidate in (root, root / "mvtec", root / "mvtec_ad"):
        if all((candidate / category).is_dir() for category in MVTEC_AD_CATEGORIES):
            return candidate
    return root


def mvtec_ad_records(
    root: Path,
    categories: Sequence[str] | None = None,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    root = _mvtec_root(Path(root))
    selected = tuple(categories or MVTEC_AD_CATEGORIES)
    unknown = sorted(set(selected) - set(MVTEC_AD_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown MVTec AD categories: {unknown}")
    train: dict[str, list[Record]] = {}
    evaluation: dict[str, list[Record]] = {}
    for category in selected:
        category_root = root / category
        good_train = _images(category_root / "train" / "good")
        if not good_train:
            raise RuntimeError(f"MVTec AD train/good is empty for {category}")
        train[category] = [
            Record(
                path=str(path), label=0, mask_path=None, category=category,
                dataset="mvtec_ad", domain="source", shift="train_good",
                role="train", mask_capability="not_evaluation",
            )
            for path in good_train
        ]
        test_root = category_root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(test_root)
        records: list[Record] = []
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            defect = defect_dir.name
            for image in _images(defect_dir):
                if defect == "good":
                    label, mask = 0, None
                else:
                    label = 1
                    mask_candidate = category_root / "ground_truth" / defect / f"{image.stem}_mask.png"
                    if not mask_candidate.is_file():
                        raise FileNotFoundError(mask_candidate)
                    mask = str(mask_candidate)
                records.append(
                    Record(
                        path=str(image), label=label, mask_path=mask,
                        category=category, dataset="mvtec_ad", domain="test",
                        shift=defect, role="evaluation", mask_capability="pixel",
                    )
                )
        if not records or {record.label for record in records} != {0, 1}:
            raise RuntimeError(f"MVTec AD evaluation support is incomplete for {category}")
        evaluation[category] = records
    return train, evaluation
