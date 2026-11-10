# AGENTS.md — live-tennis-analyzer

## What this project is

Real-time tennis video analyzer:

- RT-DETRv2 (COCO) detects players (`person`, class 0) and the tennis ball
  (`sports ball`, class 32).
- A ResNet-50 keypoint model detects 14 tennis-court points.
- A homography projects player/ball positions onto a 2D minimap.
- `main.py` reads a video file or MJPEG stream, runs both TensorRT engines,
  draws the annotated video (including court keypoints), and writes an MP4.

## Environment

- Use the project virtualenv: `source .venv/bin/activate` (full Linux) or the
  `tennis` conda env (WSL). Do not use system Python.
- Verified stack: Python 3.11, `torch==2.7.1+cu128`,
  `torchvision==0.22.1+cu128`, `tensorrt==10.8.0.43`, ONNX 1.23,
  headless OpenCV 5.
- TensorRT 10.13+ installs CUDA-13 wheels; this project is pinned to the
  CUDA-12 line (`tensorrt==10.8.0.43`) to match PyTorch cu128.
- Do **not** reintroduce PyCUDA. `utils/trt_prediction.py` uses PyTorch CUDA
  memory and streams. No system CUDA toolkit or `trtexec` is required.

## First commands in a new session

```bash
source .venv/bin/activate        # or: conda activate tennis
python tools/doctor.py           # must print READY
```

If models are missing or invalid:

```bash
python tools/setup_models.py --all
# rebuild locally if doctor reports Platform specific tag mismatch:
python tools/setup_models.py --all --overwrite
```

Smoke test:

```bash
python tools/verify_models.py \
  --video videos/sample_tennis.mp4 --frame 0 --save /tmp/preview.jpg

python main.py --stream_url videos/sample_tennis.mp4 \
  --no-display --max-frames 60
```

## Important files

- `main.py` — pipeline, CLI flags, output writer.
- `perception/object_detector.py` — RT-DETR wrapper + optional IoU tracker.
- `perception/keypoint_detector.py` — court keypoints and debug overlay.
- `utils/trt_prediction.py` — TensorRT runner (PyTorch CUDA, no PyCUDA).
- `utils/rtdetr_utils.py` — preprocessing/filtering.
- `minimap/minimap.py` — court minimap + homography.
- `tools/setup_models.py` — download + ONNX export + TensorRT build.
- `tools/doctor.py` — environment/model health check.
- `tools/verify_models.py` — single-frame inference test.
- `tools/bootstrap_linux.sh` — Omarchy/Arch bootstrap.
- `SETUP_REPORT.md` — verified WSL results.
- `OMARCHY_SETUP.md` — portable Linux/Omarchy setup and OpenCode handoff.

## Rules / conventions

- TensorRT `.trt` files are **not portable** across GPUs/TensorRT versions.
  Always rebuild on the target machine; never debug a `Platform specific tag
  mismatch` by copying plans.
- On Wayland/Omarchy or headless WSL, use `--no-display`; inspect the MP4.
- The court overlay is on by default. Flags:
  `--no-keypoints`, `--no-court-lines`, `--no-court-status`.
- Keep `models/`, `.venv/`, `output*.mp4`, `__pycache__/` out of version control.
- Local files are processed sequentially (bounded queue); network streams keep
  the original real-time frame skipping.
- The keypoint model is for elevated/broadcast cameras; do not “fix” a bad fit
  by silently disabling the overlay. Use it for diagnosis.
- Model licenses: RT-DETR Apache-2.0; keypoint model research-use-only
  (`Coddieharsh/tennis-court-keypoints`, derived from
  `yastrebksv/TennisCourtDetector`).
