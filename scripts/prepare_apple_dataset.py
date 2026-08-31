"""Prepare a one-class copy of the audited apples-e4n6e/apple-detection-oce8s export."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


def source_key(path: Path) -> str:
    return path.name.split(".rf.")[0]


def select_splits(images: dict[str, list[Path]]) -> dict[str, list[Path]]:
    # Reserve source photos for test first, then validation, never for both.
    test_keys = {source_key(path) for path in images["test"]}
    valid = [path for path in images["valid"] if source_key(path) not in test_keys]
    held_out = test_keys | {source_key(path) for path in valid}
    return {
        "train": [path for path in images["train"] if source_key(path) not in held_out],
        "valid": valid,
        "test": images["test"],
    }


def merge_labels(text: str) -> str:
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1", "2"}:
            raise ValueError(f"Unexpected label in this export: {line}")
        rows.append(" ".join(["0", *fields[1:]]))
    return "\n".join(rows) + ("\n" if rows else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/apple-original"))
    parser.add_argument("--output", type=Path, default=Path("datasets/apple"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    audit = json.loads((source / "audit.json").read_text())
    if audit["errors"] or audit["cross_split_identical_pixels"]:
        raise SystemExit("Resolve the audit's image/label errors or identical-pixel leakage first.")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}; choose a fresh output directory.")
    images = {split: sorted((source / split / "images").glob("*.jpg")) for split in ("train", "valid", "test")}
    selected = select_splits(images)
    if any(not paths for paths in selected.values()):
        raise SystemExit("An evaluation or training split is empty after filtering.")

    manifest = {
        "source_url": "https://universe.roboflow.com/apples-e4n6e/apple-detection-oce8s/dataset/1",
        "source_archive_sha256": audit["archive_sha256"],
        "license_as_listed_by_publisher": "CC BY 4.0",
        "class_mapping": {"0": "apple", "1": "apple", "2": "apple"},
        "mapping_basis": "Sample boxes visually inspected; includes whole, damaged and cut apples. Not a full manual label audit.",
        "split_policy": "Keep original test; exclude its source names from valid; exclude both from train. Source key is filename before .rf.",
        "limitations": "Different filenames may hide related photos. Existing augmentations and labels are retained. No apple-free negative images are present.",
        "splits": {},
    }
    for split, paths in selected.items():
        image_dir, label_dir = output / split / "images", output / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        box_count = 0
        for path in paths:
            label = source / split / "labels" / (path.stem + ".txt")
            content = merge_labels(label.read_text())
            shutil.copy2(path, image_dir / path.name)
            (label_dir / label.name).write_text(content)
            box_count += len(content.splitlines())
        manifest["splits"][split] = {
            "images": len(paths),
            "boxes": box_count,
            "unique_source_names": len(Counter(source_key(path) for path in paths)),
            "excluded_images": len(images[split]) - len(paths),
            "files": [path.relative_to(source).as_posix() for path in paths],
        }

    for filename in ("README.dataset.txt", "README.roboflow.txt"):
        shutil.copy2(source / filename, output / filename)
    config = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["apple"],
    }
    # JSON is valid YAML; use the standard serializer to safely encode Windows paths.
    (output / "data.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({split: {k: v for k, v in info.items() if k != "files"} for split, info in manifest["splits"].items()}, indent=2))
    print(f"Training config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
