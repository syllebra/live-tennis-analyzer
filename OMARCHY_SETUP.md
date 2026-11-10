# Exporting to full Linux (Omarchy / Arch) & reusing this project in a new OpenCode session

This document is the portable follow-up to `SETUP_REPORT.md`. It assumes the
project was developed in WSL and will be moved to a full Linux install
(**Omarchy**, which is Arch Linux based) on the same NVIDIA GPU.

Target machine used for the original verification: RTX 3090, 24 GB,
driver 615.78.02, CUDA UMD 13.4, WSL2.

---

## 0. What to export and what to leave behind

Export the **source tree and tools**, but do **not** copy:

- `models/*.trt` — TensorRT plans are tied to the GPU architecture, driver,
  TensorRT version and platform. A plan built in WSL/Windows will fail on
  Omarchy with `Platform specific tag mismatch` (this happened with the old
  `../RT-DETR/models/rtdetrv2_r50vd_6x_coco_ema.trt`).
- `.venv/`, `__pycache__/`, `output*.mp4`
- optionally `models/*.onnx` and `models/*.pth` — they are portable, but the
  setup script can download/rebuild them. Copying the `.pth` files is fine if
  you want to avoid the download.

From WSL:

```bash
cd /mnt/d/XRTrain/detectors
tar \
  --exclude='live-tennis-analyzer/models' \
  --exclude='live-tennis-analyzer/.venv' \
  --exclude='live-tennis-analyzer/__pycache__' \
  --exclude='live-tennis-analyzer/*/__pycache__' \
  --exclude='live-tennis-analyzer/output*.mp4' \
  -czf /mnt/c/Users/sylle/live-tennis-analyzer-export.tar.gz \
  live-tennis-analyzer
```

On Omarchy:

```bash
mkdir -p ~/projects
cd ~/projects
tar xzf ~/Downloads/live-tennis-analyzer-export.tar.gz
cd live-tennis-analyzer
```

The setup script expects the RT-DETR clone at `../RT-DETR` by default. It will
clone it automatically if it is missing, so you normally do **not** need to
copy `../RT-DETR` from Windows. If you do copy it, place it at
`~/projects/RT-DETR` (a sibling of this project).

---

## 1. Omarchy / Arch prerequisites

### 1.1 NVIDIA driver

Omarchy’s installer usually offers the NVIDIA driver. Verify it:

```bash
nvidia-smi
```

If it is missing, install the driver for your GPU and reboot:

```bash
sudo pacman -Syu
sudo pacman -S --needed nvidia nvidia-utils linux-headers
# For GPUs supported by the open kernel modules, this is also valid:
# sudo pacman -S --needed nvidia-open nvidia-utils linux-headers
sudo reboot
```

Do not install the full system CUDA toolkit just for this project. The PyTorch
and TensorRT pip wheels bring the CUDA runtime libraries they need. The only
system requirement is the NVIDIA **driver**.

If you already have `cuda`/`cudnn` from pacman and later see TensorRT library
mismatches, the simplest fix is to use the pip wheels described below and
rebuild the engines locally.

### 1.2 Basic tools

```bash
sudo pacman -S --needed git uv base-devel
```

`uv` is in the Arch repositories on current Omarchy. If it is not available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

`git` is used to clone the RT-DETR configs.

### 1.3 Wayland / GUI note

Omarchy uses Hyprland (Wayland). `cv2.imshow` is not reliable there. Use
`--no-display` and inspect the annotated MP4. The project is fully usable
headless.

---

## 2. One-command bootstrap (recommended)

```bash
cd ~/projects/live-tennis-analyzer
chmod +x tools/bootstrap_linux.sh
./tools/bootstrap_linux.sh --models
source .venv/bin/activate
python tools/doctor.py
```

What the script does:

1. Verifies `nvidia-smi` and `git` (does not install system drivers).
2. Uses `uv` to create `.venv` with Python 3.11.
3. Installs `torch==2.7.1` / `torchvision==0.22.1` CUDA 12.8 wheels.
4. Installs `requirements.txt` (headless OpenCV, ONNX, scipy, etc.).
5. Installs `tensorrt==10.8.0.43`, with a `pip` fallback if the TensorRT
   wheel-stub build fails.
6. With `--models`, runs `python tools/setup_models.py --all`.
7. Runs `tools/doctor.py`.

If `uv` is missing, the script prints the exact commands. Use
`./tools/bootstrap_linux.sh --install-uv --models` to install `uv` to
`~/.local/bin` first.

---

## 3. Manual setup (equivalent to the script)

```bash
cd ~/projects/live-tennis-analyzer

# Python 3.11 (TensorRT 10.8 wheels are most reliable on 3.10-3.12)
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch CUDA build — install this BEFORE requirements.txt
uv pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

# Everything except the fragile TensorRT wheel
grep -v -E '^[[:space:]]*tensorrt' requirements.txt > /tmp/req-no-trt.txt
uv pip install -r /tmp/req-no-trt.txt

# TensorRT 10.8 is the CUDA-12 line. 10.13+ defaults to CUDA-13 wheels.
uv pip install tensorrt==10.8.0.43
# If the PEP-517 stub build fails:
#   uv pip install pip
#   python -m pip install --no-cache-dir tensorrt==10.8.0.43

# Download weights + export ONNX + build local TensorRT engines
python tools/setup_models.py --all

# Health check
python tools/doctor.py
```

Why TensorRT 10.8? `tensorrt==10.13.3.9` resolves to `tensorrt-cu13` (CUDA 13),
while the PyTorch build used here is CUDA 12.8. Pinning 10.8 keeps one CUDA
generation across PyTorch and TensorRT. If you deliberately move everything to
CUDA 13, update PyTorch, TensorRT and rebuild the engines.

---

## 4. Model sources and generated artifacts

`tools/setup_models.py` produces:

| File | Purpose |
|---|---|
| `models/rtdetrv2_r50vd_6x_coco_ema.pth` | RT-DETRv2-L COCO EMA weights |
| `models/rtdetrv2_r50vd_6x_coco_ema.onnx` | self-contained detector ONNX |
| `models/rtdetrv2_r50vd_6x_coco_ema.trt` | FP16 TensorRT engine |
| `models/keypoint_detector.pth` | ResNet-50 14-keypoint checkpoint |
| `models/keypoint_detector.onnx` | self-contained keypoint ONNX |
| `models/keypoint_detector.trt` | FP16 TensorRT engine |
| `models/models.json` | versions, hashes, sizes, paths |

Sources:

- Detector: official RT-DETRv2-L r50 COCO EMA weights from
  `lyuwenyu/storage`; config from the RT-DETR repository.
- Keypoints: Hugging Face `Coddieharsh/tennis-court-keypoints`
  (`keypoints_model_geoaug.pth`), fine-tuned from
  `yastrebksv/TennisCourtDetector`. **Research-use-only licence.**

Variants:

```bash
python tools/setup_models.py --all --rtdetr-model r18   # smaller/faster
python tools/setup_models.py --all --rtdetr-model r34
python tools/setup_models.py --all --rtdetr-model r101
```

Rebuild engines after changing GPU, driver or TensorRT:

```bash
python tools/setup_models.py --all --overwrite
```

---

## 5. Running

```bash
source .venv/bin/activate

# Short local file
python main.py --stream_url videos/sample_tennis.mp4 --no-display

# Your own video
python main.py --stream_url "/path/to/video.mp4" --no-display

# Stop after 100 processed frames
python main.py --stream_url "/path/to/video.mp4" --no-display --max-frames 100

# MJPEG stream (one terminal for the server)
python streaming/start_mjpeg_stream.py "/path/to/video.mp4"
python main.py --stream_url http://127.0.0.1:8080/stream --no-display
```

### Court keypoint overlay (troubleshooting)

The output video now draws the **14 terrain/court keypoints**, the court
skeleton, and a court-fit status line. This is on by default.

- Green numbered points: outer court corners/lines (0–3)
- Orange numbered points: singles sidelines (4–7)
- Magenta numbered points: service lines and centre service line (8–13)
- Yellow lines: the inferred court skeleton
- Bottom-left status: `court keypoints: OK` or `court keypoints: FIT FAILED`

Flags:

```bash
--no-keypoints       # disable keypoints + skeleton + status drawing
--no-court-lines     # draw numbered points only
--no-court-status    # hide the OK/FAILED status text
```

How to use it for debugging:

1. Run a short clip with `--max-frames 100`.
2. Open the output MP4 and check that numbered points sit on the real line
   intersections.
3. If points drift or a corner is in the stands, the court fit is bad. The
   status line switches to `FIT FAILED` when `cv2.findHomography` cannot find a
   valid mapping; the minimap is then skipped instead of crashing.
4. For a single-frame still, run:

   ```bash
   python tools/verify_models.py \
     --video "/path/to/video.mp4" --frame 0 --save /tmp/keypoints.jpg
   ```

5. If the fit is consistently bad only on low/ground-level or heavily cropped
   cameras, that is a keypoint-model limitation (trained on elevated broadcast
   footage). Try an elevated camera or a different clip.

---

## 6. New OpenCode session on Omarchy

Everything needed is in the repository. In a fresh shell:

```bash
cd ~/projects/live-tennis-analyzer
source .venv/bin/activate
python tools/doctor.py
```

`AGENTS.md` in the project root gives a new OpenCode session the essential
context automatically. If you want to be explicit, paste this prompt:

```text
You are in ~/projects/live-tennis-analyzer on Omarchy (Arch Linux).
This project was exported from WSL; read SETUP_REPORT.md and OMARCHY_SETUP.md first.

Rules:
- Use the .venv Python 3.11 environment, not system Python.
- PyTorch is 2.7.1+cu128; TensorRT is pinned to 10.8.0.43 (CUDA 12).
- Do NOT use PyCUDA; utils/trt_prediction.py uses PyTorch CUDA streams/memory.
- TensorRT .trt files are not portable between machines; rebuild them with
  `python tools/setup_models.py --all --overwrite` if doctor.py reports INVALID.
- Use --no-display on Wayland/headless.

Start by running:
    source .venv/bin/activate
    python tools/doctor.py

If models are missing or invalid:
    python tools/setup_models.py --all

Then verify and run:
    python tools/verify_models.py --video videos/sample_tennis.mp4 --save /tmp/preview.jpg
    python main.py --stream_url videos/sample_tennis.mp4 --no-display --max-frames 60
```

Expected `doctor.py` result on a healthy machine:

```text
torch 2.7.1+cu128 ... torch CUDA available: True
object    rtdetrv2_r50vd_6x_coco_ema.trt: OK
keypoint  keypoint_detector.trt: OK
READY
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi: command not found` | Install/repair the NVIDIA driver (`nvidia` or `nvidia-open`, `nvidia-utils`), reboot |
| `torch.cuda.is_available()` is `False` | Wrong PyTorch wheel or driver not loaded. Reinstall the cu128 wheel, reboot, check `nvidia-smi` |
| `import tensorrt` fails with `libcudart.so.12` | Reinstall `tensorrt==10.8.0.43`; ensure `nvidia-cuda-runtime-cu12` is installed. Last resort: add the venv’s `nvidia/*/lib` dirs to `LD_LIBRARY_PATH` |
| Engine deserialization fails / `Platform specific tag mismatch` | Expected after copying a `.trt` from another machine. Rebuild with `python tools/setup_models.py --all --overwrite` |
| `No module named faster_coco_eval` | `uv pip install faster-coco-eval` (needed by RT-DETR `src/data`) |
| `uv pip install tensorrt` build fails | Use `python -m pip install --no-cache-dir tensorrt==10.8.0.43` inside the venv |
| `cv2.imshow` fails on Omarchy/Wayland | Use `--no-display`; inspect the MP4. For GUI, install `opencv-python` and run under XWayland |
| Court keypoints look wrong | Check the overlay; use an elevated/broadcast camera. Re-run `verify_models.py` on a single frame. The keypoint model is not trained for ground-level cameras |
| Output video is black/empty | Ensure the source opens with `cv2.VideoCapture`; check `ffmpeg` support. The output is always sized from the first frame |
| Slow FPS | RT-DETR r50 is the heaviest option. Build r18/r34 for speed: `python tools/setup_models.py --all --rtdetr-model r18 --overwrite` |

---

## 8. Expected performance

On the verified RTX 3090, 1920×1080, FP16 TensorRT:

- detector + keypoint + minimap + MP4 writing: **~17–22 FPS**
- full 929-frame broadcast clip: 53.7 s (17.3 FPS overall, 18.0 steady)
- engine builds: RT-DETR r50 ≈ 3.5 min; keypoint ≈ 36 s

Other GPUs will be slower. TensorRT logs and `models/models.json` record the
exact versions used for a build.

---

## 9. Licence / attribution

- RT-DETR: Apache-2.0 (upstream `lyuwenyu/RT-DETR`).
- Keypoint model: research-use-only, derived from
  `yastrebksv/TennisCourtDetector` and published by
  `Coddieharsh/tennis-court-keypoints`. Check the upstream terms before
  commercial use.
