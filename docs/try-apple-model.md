# Try the Trained Apple Detector

Run all commands from the repository folder. This uses `models/apple/best.pt`,
the trained checkpoint included in Git. Inference does not update the model:
showing it a new photo is not training it on that photo.

## New Photos on Windows

Put JPG or PNG files into `new-photos/` in the project folder. This folder is
ignored by Git. Use photos that were not in any dataset split, preferably your
own phone photos. Include different backgrounds, lighting, distances, and some
scenes with no apples. Similar objects are useful for checking false alarms.

```powershell
.\.venv\Scripts\yolo.exe detect predict model=models/apple/best.pt source="new-photos" device=0 conf=0.5 save=True
```

For one image, replace `source="new-photos"` with `source="C:/path/to/photo.jpg"`.
Keep quotes around paths containing spaces. The command saves annotated copies;
it does not overwrite the input photos. Open the results directory printed at the
end of the command. Prediction runs normally get numbered folders on repeat runs.

Predictions alone do not produce an accuracy score: new photos need reviewed
ground-truth boxes before you can calculate precision, recall, and mAP. Do not
use the reserved test split to repeatedly adjust thresholds or training settings.

## Get the Update on Mac

In your existing Mac checkout:

```bash
git pull --ff-only origin main
```

If Git reports local changes or diverged history, preserve those changes and
resolve them before proceeding; do not reset or force-pull.

For a fresh checkout instead:

```bash
git clone https://github.com/ivankao83/Computer-Vision---Object-Recognition-.git
cd Computer-Vision---Object-Recognition-
```

The checkpoint comes with the pull/clone. Do not copy the Windows `.venv` folder:
its Python executables and dependencies are not Mac-compatible. The training
images and `runs/` directories are unnecessary for inference and are not in Git.

Use Python 3.11 or 3.12 for this workflow, then create a Mac environment. If a
working Mac `.venv` already exists, skip its creation and activate it instead.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training]"
```

Do not run the Windows CUDA installation command on Mac. Python/package support
depends on the Mac architecture and macOS version; if pip cannot find compatible
PyTorch wheels, record `uname -m` and `python3 --version` before troubleshooting.

## Mac Webcam

```bash
python -m edge_object_recognition.app --weights models/apple/best.pt --camera 0 --device cpu
```

Allow camera access for Terminal or the app running your terminal. Check **System
Settings > Privacy & Security > Camera** if access was denied. Close other apps
using the webcam. If camera zero is not the desired camera, try `--camera 1`.

- `q` or Escape: quit.
- `s`: save an annotated screenshot in `captures/`.
- `--confidence 0.6`: use a stricter detection threshold (may miss more apples).

The CPU setting is the initial compatibility check. On Apple Silicon, check:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If it prints `True`, you can try `--device mps` instead of `--device cpu`.
Measure performance rather than assuming it will match the RTX 3070.

## New Photos on Mac

With the Mac virtual environment activated:

```bash
yolo detect predict model=models/apple/best.pt source="new-photos" device=cpu conf=0.5 save=True
```

Photos can be transferred from your phone using your normal photo-transfer method.
Check photos for private information before including any in a public demo.

## References

- [Ultralytics prediction inputs and options](https://docs.ultralytics.com/modes/predict/)
- [Apple camera access settings](https://support.apple.com/guide/mac-help/mchlf6d108da/mac)
