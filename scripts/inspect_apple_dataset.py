"""Inspect the downloaded apple archive without trusting its metadata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/apple-original"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise SystemExit(f"Refusing to overwrite {root}; choose a fresh --output directory.")

    with zipfile.ZipFile(args.archive) as archive:
        entries = {}
        repeated_entries = Counter()
        for entry in archive.infolist():
            name = entry.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe archive path: {name}")
            if entry.is_dir():
                continue
            if name in entries:
                if archive.read(entry) != archive.read(entries[name]):
                    raise ValueError(f"Conflicting duplicate archive entry: {name}")
                repeated_entries[name] += 1
            entries[name] = entry
        root.mkdir(parents=True)
        for name, entry in entries.items():
            target = root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))

    counts = {}
    errors = []
    hashes = defaultdict(list)
    source_names = defaultdict(list)
    candidates = {}
    for split in ("train", "valid", "test"):
        images = sorted((root / split / "images").glob("*.jpg"))
        classes = Counter()
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                errors.append(f"Unreadable image: {path.relative_to(root)}")
                continue
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256(image.tobytes() + str(image.shape).encode()).hexdigest()
            hashes[digest].append(relative)
            source_names[path.name.split(".rf.")[0]].append(relative)
            label = root / split / "labels" / (path.stem + ".txt")
            if not label.exists():
                errors.append(f"Missing labels: {relative}")
                continue
            rows = []
            for line in label.read_text().splitlines():
                try:
                    values = [float(value) for value in line.split()]
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError("invalid row")
                    class_id, x, y, width, height = values
                    if class_id not in (0, 1, 2) or not (0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError("invalid class or center")
                    if not (0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("invalid box size")
                    rows.append(values)
                    classes[int(class_id)] += 1
                except ValueError:
                    errors.append(f"Invalid label: {label.relative_to(root)}: {line}")
            key = (split, tuple(sorted({int(row[0]) for row in rows})))
            candidates.setdefault(key, (path, rows))
        counts[split] = {"images": len(images), "boxes_by_class": dict(classes)}

    cross_split = lambda paths: len({p.split('/')[0] for p in paths}) > 1
    report = {
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "splits": counts,
        "repeated_archive_entries": dict(repeated_entries),
        "errors": errors,
        "cross_split_identical_pixels": [paths for paths in hashes.values() if cross_split(paths)],
        "cross_split_source_names": [paths for paths in source_names.values() if cross_split(paths)],
    }
    report_path = root / "audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    colors = [(0, 210, 255), (255, 130, 0), (180, 0, 255)]
    tiles = []
    for (split, ids), (path, rows) in candidates.items():
        tile = cv2.resize(cv2.imread(str(path)), (360, 360))
        for class_id, x, y, width, height in rows:
            left, top = int((x - width / 2) * 360), int((y - height / 2) * 360)
            right, bottom = int((x + width / 2) * 360), int((y + height / 2) * 360)
            color = colors[int(class_id)]
            cv2.rectangle(tile, (left, top), (right, bottom), color, 2)
            cv2.putText(tile, str(int(class_id)), (left, max(15, top)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, f"{split}: class IDs {ids}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        if not cv2.imwrite(str(root / "label-preview.jpg"), sheet):
            raise OSError("Could not save label preview")
    print(json.dumps({key: value if key == "splits" else len(value) for key, value in report.items() if key != "archive_sha256"}, indent=2))
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
