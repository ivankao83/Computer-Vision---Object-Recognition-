from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
}


def main() -> int:
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    for filename, url in FILES.items():
        target = output_dir / filename
        if target.exists():
            print(f"Already exists: {target}")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, target)
        print(f"Saved {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

