# Apple Dataset Preparation

## Source and Attribution

Dataset: **Apple detection**, version 1, published by **apples** on
[Roboflow Universe](https://universe.roboflow.com/apples-e4n6e/apple-detection-oce8s/dataset/1).
The publisher lists **CC BY 4.0**. Retain the source README files and attribution
when sharing a derivative; check the source license before publication.

Original ZIP: `Apple detection.v1i.yolov8.zip`.
SHA-256: `a2bdab333d843fab90d92b08cdd4216b916789c219598a3b11ea1273e32bbbc2`.
This is a public dataset, not photos collected or originally annotated by this project.

## Audit Findings

- The archive actually has 2,215 JPG images, despite its README claiming 2,877.
- All images decode and have matching labels; all label rows pass basic numeric checks.
- `data.yaml` appears three times with identical contents and contains export text
  as class names. Training labels use only class 0, while evaluation uses 0, 1, and 2.
- Visually checked sample boxes for all three IDs mark apples, including damaged
  and cut apples. This is a sample review, not proof that every annotation is correct.
- 101 source filename groups occur in multiple splits. Sample images confirm that
  transformed versions of the same scene can appear in training and evaluation.
- No byte-identical decoded images were found across splits. That does not rule out
  near-duplicates, as the repeated scenes show.

## Changes Made

The original download remains untouched. Extraction is in `datasets/apple-original/`.
The derived copy is in `datasets/apple/`. All three IDs map to `0: apple`, preserving
the bounding-box coordinates. This pilot class includes whole, damaged, and cut apples.

Images keep their original evaluation splits, except for overlapping source names:
test takes priority over validation, and both take priority over training. The
source key is the filename before `.rf.`. We excluded 478 training images and
9 validation images from the derived copy. We did not delete any source images.

| Split | Images | Source Filename Groups | Apple Boxes |
| --- | ---: | ---: | ---: |
| Training | 1,447 | 293 | 6,595 |
| Validation | 240 | 221 | 1,055 |
| Test | 41 | 39 | 186 |

The script writes `data.yaml` as JSON syntax, which is also valid YAML. Its absolute
dataset path is machine-specific. On another computer, rerun preparation there.
`preparation.json` records the exact retained file lists, counts, mapping, and source hash.

## Reproduce

From the repository folder in Windows PowerShell, with the base dependencies installed:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_apple_dataset.py "C:\Users\ivank\Downloads\Apple detection.v1i.yolov8.zip"
.\.venv\Scripts\python.exe scripts/prepare_apple_dataset.py
```

These commands have already been run on this computer. They refuse to overwrite
existing output. To repeat, choose fresh directories with `--output` and pass the
matching extracted directory as `--source` to preparation. On Mac, use your virtual
environment's `python` and your local path to the ZIP.

## Train Next

An initial training run and test evaluation are now complete; see the
[model card](../models/apple/README.md). To reproduce training on another setup,
install the optional training tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[training]"
```

For a CPU-only training run, which can be slow:

```powershell
.\.venv\Scripts\yolo.exe detect train model=yolo11n.pt data=datasets/apple/data.yaml epochs=50 imgsz=640 batch=8 device=cpu workers=0 seed=42 project=runs/detect name=apple
```

This fine-tunes pretrained weights; it does not train from random initialization.
The machine has an NVIDIA RTX 3070 with 8 GB memory, but GPU training requires
CUDA-enabled PyTorch. Verify `torch.cuda.is_available()` before switching to
`device=0`. Having an NVIDIA card alone does not confirm that the Python environment
can use it. See the general training guide for evaluation and custom inference.

## Limits for a Portfolio

Known filename overlap was removed, but photos with unrelated names may still
depict the same scene. Existing augmentations remain. There are no apple-free
negative images; add reviewed negative scenes and test false positives before
claiming robust detection. The test set is small and lacks documented capture sessions.

Use this as an initial experiment. Add independently collected test photos for a
stronger real-world evaluation. Report that you **audited and prepared a public apple
dataset**, then trained and evaluated a model only after those steps are complete.
Do not describe this dataset as original photography or the prepared split as fully
free of duplicate scenes.

Datasets are ignored by Git. Preserve the ZIP and prepared data in a separate backup.
