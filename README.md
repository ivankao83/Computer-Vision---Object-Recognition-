# Edge Object Recognition

A lightweight edge object-recognition pipeline for a webcam or Raspberry Pi camera.

Pipeline:

```text
Camera -> OpenCV frames -> ML model -> bounding box + classification + FPS
```

The starter implementation uses OpenCV DNN with MobileNet-SSD. It shows live detections, confidence scores, and frame rate, which matches the basic project shown in the reference image.

## Trained Apple Detector

The repository includes the fine-tuned checkpoint at `models/apple/best.pt` (about
5.5 MB). You do not need the training dataset or another training run to use it.
See [new-photo and Mac webcam instructions](docs/try-apple-model.md) and the
[model card](models/apple/README.md) for results, limitations, and attribution.

On a Mac, from the repository folder (use a fresh Mac virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[training]"
python -m edge_object_recognition.app --weights models/apple/best.pt --device cpu
```

Allow camera access when prompted. Press `q` to quit. The CPU command does not
require NVIDIA/CUDA; Apple Silicon users can also try `--device mps`.

## Features

- Live camera inference from a USB webcam or Raspberry Pi camera exposed as a video device
- Bounding boxes and class labels drawn on the video feed
- FPS counter and current top detection panel
- Configurable confidence threshold, camera index, and frame size
- Small codebase that can be extended to custom objects later

## Project Layout

```text
.
├── models/                         # downloaded model files go here
├── scripts/download_mobilenet_ssd.py
├── src/edge_object_recognition/
│   ├── app.py                      # command-line entry point
│   ├── camera.py                   # camera wrapper
│   ├── detector.py                 # OpenCV DNN detector
│   ├── labels.py                   # class label handling
│   └── ui.py                       # drawing overlays
└── tests/
```

## Setup

Install Python 3.9+ first if `python --version` does not work in your terminal.

The MobileNet-SSD Caffe model requires OpenCV 4.x. OpenCV 5 removed the
`readNetFromCaffe` loader, so the dependencies constrain OpenCV to versions below 5.

Create a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download the starter model:

```powershell
python scripts/download_mobilenet_ssd.py
```

Run live detection:

```powershell
python -m edge_object_recognition.app --model-dir models
```

If you have multiple cameras, try:

```powershell
python -m edge_object_recognition.app --camera 1
```

## Raspberry Pi Notes

Install camera and OpenCV dependencies:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2
```

For a USB camera:

```bash
python -m edge_object_recognition.app --camera 0 --width 640 --height 480
```

For a Pi Camera Module, enable the camera interface and verify that it is available through libcamera:

```bash
libcamera-hello
```

Many Pi setups expose the camera through `/dev/video0`. If yours does not, use `libcamera-vid` or Picamera2 to bridge frames into OpenCV before calling the detector.

## Train Your Own Objects

Follow [the custom training guide](docs/custom-training.md) to collect your own
dataset, draw bounding boxes, train YOLO11n, evaluate it, and use the trained
weights in this app. Both fine-tuning and training from random weights are covered.

For the downloaded apple dataset, see [the apple audit and setup](docs/apple-dataset.md).

Capture raw training images (no model download needed):

```bash
python -m edge_object_recognition.collect
```

Run your own trained detection model after installing `pip install -e ".[training]"`:

```bash
python -m edge_object_recognition.app --weights runs/detect/objects/weights/best.pt
```

Custom training is optional; the original MobileNet-SSD command still works
without installing Ultralytics. Training images are not included. The apple
checkpoint is included; other model files and training outputs remain ignored.

## Keyboard Controls

- `q` or `Esc`: quit
- `s`: save the current annotated frame to `captures/`
