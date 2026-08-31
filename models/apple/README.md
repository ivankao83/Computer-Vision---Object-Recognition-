# Apple Detector Model Card

`best.pt` is the locally fine-tuned YOLO11n detection checkpoint. It is included
so the same model can run on another computer without repeating training.
Only load PyTorch checkpoints you trust.

## Training

- Base: Ultralytics `yolo11n.pt` pretrained weights; this is fine-tuning, not training from scratch.
- Runtime: Ultralytics 8.4.135, PyTorch 2.10.0+cu128, Python 3.12.13.
- Hardware: NVIDIA RTX 3070, 8 GB VRAM.
- Configuration: 50 epochs, 640 input size, batch 8, seed 42, workers 0,
  automatic optimizer selection (AdamW), AMP enabled, other training defaults.
- Training time reported: 0.612 hours (about 37 minutes).
- Classes: one, `apple`, including whole, damaged, and cut apples.
- Checkpoint: selected using validation performance, then evaluated on test.

Training command (the original run):

```text
yolo detect train model=yolo11n.pt data=datasets/apple/data.yaml epochs=50 imgsz=640 batch=8 device=0 workers=0 seed=42 project=runs/detect name=apple
```

The original run saved under `runs/detect/runs/detect/apple/`. The bundled
checkpoint is a byte-for-byte copy of that run's `weights/best.pt`.

## Reported Results

| Split | Images | Labeled Apples | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 240 | 1,055 | 0.832 | 0.842 | 0.874 | 0.533 |
| Test | 41 | 186 | 0.865 | 0.898 | 0.938 | 0.630 |

These values come from the completed training and test-evaluation console reports.
Test inference was reported at 6.5 ms/image on the RTX 3070; this is batched
model inference timing, not webcam FPS or a Mac performance measurement.
mAP50 is not the percentage accuracy on arbitrary future photos.

## Dataset and Limitations

Public dataset: **Apple detection**, version 1, by **apples** on
[Roboflow Universe](https://universe.roboflow.com/apples-e4n6e/apple-detection-oce8s/dataset/1).
The publisher lists CC BY 4.0. This project did not collect or originally annotate
the photos. Changes: merged malformed class IDs into `apple`, removed known
source-name overlap between splits, and generated a local path configuration.
See [the dataset audit](../../docs/apple-dataset.md) and preparation scripts.

The training set has 1,447 images, many augmented. Original images were stretched
to 640 x 640; some are blurry or have visible patch artifacts. The test set is
small and has no apple-free backgrounds. Different filenames may still conceal
related images even after the known overlap was removed. These are pilot results,
not evidence of reliable deployment on all backgrounds or cameras.

## Upstream License

The checkpoint is derived from Ultralytics YOLO11 weights. Ultralytics publishes
its open-source code and models under AGPL-3.0; see the included `LICENSE` and
[upstream source](https://github.com/ultralytics/ultralytics/tree/v8.4.135).
The model architecture, training procedure, and usage are documented above.
Dataset attribution and licensing are separate from the model/software license.

See [usage instructions](../../docs/try-apple-model.md) for Windows photos and the Mac webcam.
