# Your Own Object Detection Dataset

Your "database" is a versioned collection of photos and bounding-box labels.
The model learns from those examples; it does not look up photos at runtime.
This repository now supports collection and custom inference, but you still need
to supply images, label them, train, and measure the results.

## 1. Choose a Small, Useful Task

Start with 3-5 classes you can photograph yourself. Examples: desk objects,
electronics components, or recyclable packaging. Define what each class includes.
Decide whether you want to recognize a category (any mug) or one specific item
(your particular mug). Category recognition needs different examples of that category.

A practical pilot target is 100-300 varied images per class, not a guarantee of
accuracy. Expand based on validation failures. Photograph different backgrounds,
lighting, distances, angles, occlusions, and several objects together. Include
some scenes containing none of your target objects. Avoid collecting only one
object against one background or hundreds of nearly identical video frames.

## 2. Set Up

From the repository folder, activate your existing virtual environment. On Mac:

```bash
source .venv/bin/activate
python -m pip install -e ".[training]"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[training]"
```

If you do not have a virtual environment yet, follow the main README setup first.
This optional install adds Ultralytics and PyTorch and can be large. Train on your
computer first; measure and optimize Raspberry Pi inference separately.

## 3. Collect Raw Photos

```bash
python -m edge_object_recognition.collect --camera 0
```

Press `s` to save a photo, and `q` or Escape to quit. Each run writes to a new
session directory under `datasets/raw/`. These images have no drawn detections.
Do not use the original app's annotated screenshots as training images.
Phone photos are also fine; keep them organized by capture session.

## 4. Label and Split

Use a bounding-box annotation tool that exports **YOLO detection** format.
For every image, draw a tight rectangle around every visible target object and
assign its class. A folder named "mug" alone is not enough for object detection.
Keep the class order identical across annotation sessions and dataset splits.

Set aside approximately 70% for training, 20% for validation, and 10% for testing.
Split by capture session or scene BEFORE augmentation. Near-duplicate frames must
not cross splits. Keep examples of every class in each split; collect extra
sessions when needed. Reserve different physical items too if you want to measure
generalization to unseen items. Never tune on your final test set.

Export and organize matched files like this:

```text
datasets/objects/
  images/
    train/example.jpg
    val/another.jpg
    test/held-out.jpg
  labels/
    train/example.txt
    val/another.txt
    test/held-out.txt
```

Each label file uses the image's basename. Each object has one row:

```text
class_id center_x center_y width height
```

Class IDs start at zero. Coordinates are normalized by image width/height to
values between 0 and 1. For example, `0 0.5 0.5 0.2 0.4` marks class zero at the
image center. Use an empty label file for a reviewed image with no target objects.
An unreviewed image must not silently become a negative example.

Copy `configs/objects.example.yaml` to `configs/objects.yaml`. Edit its `path` to
the absolute dataset directory on your machine and replace `names` with your
actual label mapping. On Windows, use forward slashes in YAML paths, such as
`C:/Users/ivank/datasets/objects`. On Mac, use `/Users/yourname/...`.
The example YAML contains placeholder classes, not an actual dataset.

Record dataset version, image counts per split/class, collection conditions,
labeling rules, and the exact file lists belonging to each split.

## 5. Train

Start with fine-tuning, which trains pretrained YOLO11n weights on YOUR labeled
photos. This is legitimate custom-model training. Describe it as fine-tuning,
not as inventing the architecture or training from scratch.

```bash
yolo detect train model=yolo11n.pt data=configs/objects.yaml epochs=50 imgsz=640 batch=8 device=cpu workers=0 seed=42 project=runs/detect name=objects
```

The initial weights download requires internet. `device=cpu` works without a GPU
but can be slow. On supported Apple Silicon Macs, try `device=mps`; on a configured
NVIDIA CUDA machine, use `device=0`. Reduce `batch` to 4 or 2 if memory runs out.
Fifty epochs is a starting experiment, not a promised accuracy or training time.

To train from randomly initialized weights instead, use a model architecture
YAML and explicitly disable pretrained weights:

```bash
yolo detect train model=yolo11n.yaml pretrained=False data=configs/objects.yaml epochs=100 imgsz=640 batch=8 device=cpu workers=0 seed=42 project=runs/detect name=objects-scratch
```

This normally needs more data and compute. Compare the two runs on the same held-out
data. Both use an existing YOLO architecture and training implementation.

The first fine-tuning run saves to `runs/detect/objects/`; repeated runs may get
numbered directories. Use the actual directory printed by training. Keep its
configuration, results CSV, curves, and `weights/best.pt`. A fixed seed helps
repeatability but does not guarantee identical results across devices.

## 6. Evaluate and Run

Use validation results to choose settings. Once those choices are frozen, evaluate
the final model on the reserved test split:

```bash
yolo detect val model=runs/detect/objects/weights/best.pt data=configs/objects.yaml split=test device=cpu workers=0 project=runs/detect name=objects-test
```

Report precision (how many detections were correct), recall (how many targets were
found), and mAP50-95 (detection quality across overlap thresholds), including per-class
failures. A confidence score is not overall accuracy. Keep test results separate
from validation scores used to select `best.pt`.

Use your weights in the existing webcam interface:

```bash
python -m edge_object_recognition.app --weights runs/detect/objects/weights/best.pt --device cpu --confidence 0.5
```

Class names come from the trained weights. Only load weights you created or trust.
The existing app's `s` still saves an annotated demo image; `q` quits. Record FPS
with hardware, frame size, and model input size. The overlay measures whole-loop
throughput, not isolated neural-network latency. No Pi performance is promised.

## 7. Portfolio Evidence

Include your problem statement, dataset collection and labeling process, split
strategy, training configuration, evaluation plots, mistakes, and a short live demo.
Compare fine-tuning versus random initialization, or two dataset versions, under
the same evaluation conditions. Explain what improved and what still fails.

Use wording such as: "Collected and annotated my own X-class dataset, fine-tuned
YOLO11n, evaluated on held-out capture sessions, and integrated live inference."
Replace X and add measured metrics only after you have results.

`datasets/`, `runs/`, and model weights are ignored by Git to avoid accidentally
publishing large files or private photos. Git push does NOT back these up: keep a
separate backup to move them between computers. Commit code, label definitions,
dataset documentation, and a small reviewed set of demo images instead. Check
photos for faces, addresses, and screens before sharing. Link any published dataset
or model release and document applicable third-party licenses.

## References

- [YOLO11 models](https://docs.ultralytics.com/models/yolo11/)
- [Training options and Apple Silicon support](https://docs.ultralytics.com/modes/train/)
- [YOLO detection dataset format](https://docs.ultralytics.com/datasets/detect/)
- [Prediction results API](https://docs.ultralytics.com/modes/predict/)
