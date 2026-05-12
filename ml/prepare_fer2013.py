"""
prepare_fer2013.py — ממיר FER2013 מ-CSV לתיקיות תמונות לפי רגש.

מקורות נתמכים:
    data/fer2013/_download/train.csv.zip
    data/fer2013/_download/test.csv.zip   (ללא תוויות — לא בשימוש)
    data/fer2013/_download/fer2013/fer2013.csv

פלט:
    data/fer2013/train/<emotion>/*.png
    data/fer2013/test/<emotion>/*.png
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "test",
    "PrivateTest": "test",
}


def _extract_csv(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        csv_name = archive.namelist()[0]
        target = zip_path.with_suffix("")
        if not target.exists():
            archive.extract(csv_name, path=zip_path.parent)
            extracted = zip_path.parent / csv_name
            if extracted != target:
                extracted.rename(target)
        return target


def _pixels_to_image(pixels: str) -> Image.Image:
    values = np.asarray(pixels.split(), dtype=np.uint8).reshape(48, 48)
    return Image.fromarray(values, mode="L")


def _export_labeled_csv(csv_path: Path, output_root: Path, split_name: str) -> int:
    saved = 0
    counters = {name: 0 for name in EMOTION_NAMES}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "emotion" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain an emotion column.")

        for row in reader:
            label = int(row["emotion"])
            emotion = EMOTION_NAMES[label]
            counters[emotion] += 1
            image = _pixels_to_image(row["pixels"])

            out_dir = output_root / split_name / emotion
            out_dir.mkdir(parents=True, exist_ok=True)
            image.save(out_dir / f"{emotion}_{counters[emotion]:05d}.png")
            saved += 1

            if saved % 2000 == 0:
                print(f"  {split_name}: saved {saved} images", flush=True)

    return saved


def _export_master_csv(csv_path: Path, output_root: Path, splits: list[str]) -> dict[str, int]:
    counts = {"train": 0, "test": 0}
    counters = {
        "train": {name: 0 for name in EMOTION_NAMES},
        "test": {name: 0 for name in EMOTION_NAMES},
    }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_name = USAGE_TO_SPLIT.get(row["Usage"])
            if split_name is None or split_name not in splits:
                continue

            label = int(row["emotion"])
            emotion = EMOTION_NAMES[label]
            counters[split_name][emotion] += 1
            image = _pixels_to_image(row["pixels"])

            out_dir = output_root / split_name / emotion
            out_dir.mkdir(parents=True, exist_ok=True)
            image.save(out_dir / f"{emotion}_{counters[split_name][emotion]:05d}.png")
            counts[split_name] += 1

            if counts[split_name] % 2000 == 0:
                print(f"  {split_name}: saved {counts[split_name]} images", flush=True)

    return counts


def prepare(download_dir: Path, output_dir: Path, splits: list[str]) -> None:
    master_csv = download_dir / "fer2013" / "fer2013.csv"
    train_zip = download_dir / "train.csv.zip"
    test_zip = download_dir / "test.csv.zip"

    if master_csv.exists():
        print(f"Exporting from master CSV: {master_csv}")
        counts = _export_master_csv(master_csv, output_dir, splits)
        print(f"Done: train={counts['train']}, test={counts['test']}, output={output_dir}")
        return

    if "train" in splits:
        if not train_zip.exists():
            raise FileNotFoundError(f"Missing train archive: {train_zip}")
        train_csv = _extract_csv(train_zip)
        print(f"Exporting train split from {train_csv}")
        train_count = _export_labeled_csv(train_csv, output_dir, "train")
        print(f"Train images saved: {train_count}")

    if "test" in splits:
        raise FileNotFoundError(
            "Labeled test split is unavailable from test.csv in this mirror. "
            f"Extract {master_csv} or download fer2013.tar.gz into {download_dir}."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FER2013 image folders from CSV archives.")
    parser.add_argument("--download-dir", default="data/fer2013/_download")
    parser.add_argument("--output-dir", default="data/fer2013")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(Path(args.download_dir), Path(args.output_dir), list(args.splits))
