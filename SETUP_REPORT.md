# Tennis Analyzer — WSL Setup & Verification Report

Date: 2026-11-05  
Host: WSL2, NVIDIA GeForce RTX 3090 (24 GB), driver 615.78.02 / CUDA UMD 13.4  
Working tree: `/mnt/d/XRTrain/detectors/live-tennis-analyzer`

## 1. Result

The analyzer now runs end-to-end on the GPU:

```bash
conda activate tennis
python main.py --stream_url <video-or-stream> --no-display
```

Verified on:

- Virtual Tennis clips (`a.mp4`, `b.mp4`)
- Real ATP broadcast clips (`tennis-tracking/VideoInput/video_input2..8.mp4`)
- MJPEG streaming mode via `streaming/start_mjpeg_stream.py`

Steady-state throughput was **~17–22 FPS at 1920×1080** with RT-DETRv2-L (r50)
FP16 TensorRT + ResNet-50 keypoint TensorRT + minimap drawing + MP4 writing.

Output videos now also draw the **14 court keypoints, court skeleton and a
court-fit status line** for troubleshooting terrain/court detection.

## 2. Why the original setup failed

1. **`tools/export_onnx.py` was not Python.** It was a 500+ line saved GitHub
   HTML page. Any attempt to run it as a Python exporter failed immediately.
2. **`../RT-DETR/models/*.yml` were also saved HTML pages** for r18/r50, not
   YAML configs. Only `rtdetrv2_r101vd_6x_coco.yml` was valid text.
3. The existing `../RT-DETR/models/rtdetrv2_r50vd_6x_coco_ema.onnx` was split
   into `.onnx` + `.onnx.data` (external data). That is easy to mishandle and
   cannot be used as a simple single-file artifact by every TensorRT setup.
4. The existing `../RT-DETR/models/rtdetrv2_r50vd_6x_coco_ema.trt` (217 MB)
   **cannot be loaded in WSL**. Deserializing it with TensorRT 10.8 fails with
   `Serialization assertion plan->header.pad == expectedPlatformTag failed.
   Platform specific tag mismatch detected.` It was built for a different
   platform (very likely Windows), and TensorRT plans are not portable.
5. **No system CUDA toolkit / `nvcc` / `trtexec`** was present in WSL. The
   project's README assumes `trtexec`.
6. `requirements.txt` omitted `torchvision`, `PyYAML`, `onnx`, `tensorrt`,
   `scipy`, `tensorboard`, `faster-coco-eval`, and `pycuda`. PyCUDA is the
   package most likely to fail to build on Windows/WSL without CUDA headers.
7. `perception/object_detector.py` referenced an undefined `BYTETracker`, so
   `ObjectDetector.__init__` crashed even with valid models.
8. `main.py` unconditionally imported `pycuda.autoinit`, hard-coded a
   1920×1080 writer, and always opened `cv2.imshow` (fails on headless WSL).
9. The keypoint `.pth` was not included with the project.

## 3. Environment created

```text
conda env: tennis (Python 3.11.15)
torch       2.7.1+cu128
torchvision 0.22.1+cu128
TensorRT    10.8.0.43 (CUDA 12 build)
onnx        1.23.2
OpenCV      5.0.0 (headless)
NumPy       2.4.6
```

**Important:** TensorRT 10.13+ installs the CUDA-13 (`tensorrt-cu13`) wheels.
For the CUDA-12 PyTorch build used here, pin `tensorrt==10.8.0.43`.
No system CUDA toolkit or `trtexec` is needed; TensorRT engines are built with
the TensorRT Python API.

## 4. Models produced

| File | Size | SHA-256 |
|---|---:|---|
| `models/rtdetrv2_r50vd_6x_coco_ema.pth` | 165 MiB | `a1f30829515a5096702dbf2ab2e06b5b0c7c6cd96234c257d9862abe45f03641` |
| `models/rtdetrv2_r50vd_6x_coco_ema.onnx` | 161.4 MiB | `f7e87f6d8029db0109d2c968df936b31f3e60ff3f7051dd996e150895383bc22` |
| `models/rtdetrv2_r50vd_6x_coco_ema.trt` | 91.6 MiB | `b03868495922c0302bb3abcc499446e27e02d8b16ec67198b35e8fd5ef667282` |
| `models/keypoint_detector.pth` | 90.2 MiB | `d4550c7bba4135c3dea5e802c53f6e938c1405f592ed772ac7dc5d3f4e68f6a3` |
| `models/keypoint_detector.onnx` | 89.8 MiB | `5ef38af6905fd15663b676fe2ee6b31d0177aaa0c32ba744ca729076b312e751` |
| `models/keypoint_detector.trt` | 47.0 MiB | `c66ef0b37d22561954691bd027466a2f8e325effb8502831bf3954c1d6d2fd28` |

Model I/O:

```text
RT-DETR:  images (1,3,640,640) float32
          orig_target_sizes (1,2) int64
       -> labels (1,300) int64, boxes (1,300,4) float32, scores (1,300) float32

Keypoint: input (1,3,224,224) float32 -> output (1,28) float32
```

Sources:

- RT-DETRv2-L r50 COCO EMA weights: official `lyuwenyu/storage` release
  (`v0.1/rtdetrv2_r50vd_6x_coco_ema.pth`), config from the local RT-DETR clone.
- Keypoint weights: Hugging Face `Coddieharsh/tennis-court-keypoints`
  (`keypoints_model_geoaug.pth`), a ResNet-50 14-keypoint regressor fine-tuned
  from `yastrebksv/TennisCourtDetector`. **Research-use-only licence.**
  Converted from plain `resnet50.fc` layout to the repository's
  `KeypointModel` (`backbone.*` + `keypoint_head.0.*`) layout.

Build times on this machine:

- RT-DETR r50 ONNX export: ~40 s
- RT-DETR r50 FP16 TensorRT engine: 207 s
- Keypoint ONNX export: ~5 s
- Keypoint FP16 TensorRT engine: 36 s

## 5. Code changes

Added:

- `tools/setup_models.py` — one-command download + validate + ONNX export +
  TensorRT build + `models/models.json` manifest. Supports r18/r34/r50/r101,
  `--download-only`, `--skip-onnx`, `--skip-trt`, `--fp32`, `--component`.
- `tools/export_onnx.py` — replaced the accidentally committed HTML page with a
  working RT-DETR ONNX export wrapper.
- `tools/verify_models.py` — single-frame smoke test that prints detections and
  keypoints and optionally saves an annotated preview.
- `tools/doctor.py` — environment/model health check for a fresh machine or
  OpenCode session.
- `tools/bootstrap_linux.sh` — Omarchy/Arch bootstrap using `uv` + CUDA-12
  PyTorch, with a TensorRT pip fallback.
- `OMARCHY_SETUP.md` — export, Omarchy/Arch setup, troubleshooting and a new
  OpenCode-session handoff prompt.
- `AGENTS.md` — context automatically read by OpenCode in a new session.
- `.gitignore` — excludes generated models, outputs, and caches.
- `SETUP_REPORT.md` — this report.

Changed:

- `utils/trt_prediction.py` — PyCUDA removed. Device buffers and streams now use
  PyTorch (`torch.empty(..., device='cuda')`, `torch.cuda.Stream`). Same public
  API (`TRTInferenceObject`, `TRTInferenceKeypoint`).
- `main.py` — removed `pycuda.autoinit`; writer size now comes from the first
  frame; added `--output`, `--output_fps`, `--no-display`, `--max-frames`,
  `--minimap-width`; bounded frame queue; court-keypoint overlay flags.
- `perception/keypoint_detector.py` — draws all 14 numbered court keypoints,
  the court skeleton and group-coloured points (green/orange/magenta) for
  debugging.
- `streaming/stream_handler.py` — local files are processed sequentially through
  a bounded queue (no frame loss); network streams keep the original real-time
  frame skipping. Normal EOF no longer reports a fake error.
- `perception/object_detector.py` — replaced the missing `BYTETracker` with a
  small IoU tracker (the tracking path is optional and not used by `main.py`).
- `utils/rtdetr_utils.py` — fixed the second-player selection bug in
  `remove_non_players`.
- `minimap/minimap.py` — guards against `None`/singular homographies; removed
  noisy tracker prints.
- `requirements.txt` and `README.md` — updated installation and usage.

## 6. Verification runs

Commands used:

```bash
# Model preparation (already done)
python tools/setup_models.py --all

# Single-frame sanity check
python tools/verify_models.py \
  --video "/mnt/c/Users/sylle/Videos/NVIDIA/Virtual Tennis/a.mp4" \
  --frame 0 --save /tmp/verify.jpg

# Full pipeline on a local file
python main.py \
  --stream_url "/mnt/d/XRTrain/detectors/tennis-tracking/VideoInput/video_input2.mp4" \
  --no-display --max-frames 150 --output output_video_input2.mp4

# Full 168-frame clip (every frame processed)
python main.py \
  --stream_url "/mnt/d/XRTrain/detectors/tennis-tracking/VideoInput/video_input8.mp4" \
  --no-display --output output_video_input8.mp4
```

Single-frame checks across all supplied clips:

| Clip | Frame | Persons (>0.6) | Balls (>0.4) | Keypoints finite | Homography |
|---|---:|---:|---:|---|---|
| video_input2 (Vienna) | 0 / 464 | 6 / 6 | 1 / 1 | yes | OK |
| video_input3 | 0 / 46 | 9 / 9 | 1 / 1 | yes | OK |
| video_input4 | 0 / 62 | 12 / 11 | 1 / 0 | yes | OK |
| video_input5 (Rome clay) | 0 / 61 | 7 / 5 | 0 / 1 | yes | OK |
| video_input6 | 0 / 53 | 11 / 11 | 1 / 1 | yes | OK |
| video_input8 | 0 / 84 | 7 / 7 | 1 / 2 | yes | OK |
| Virtual Tennis a | 0 / 1129 | 7 / 12 | 1 / 0 | yes | OK |
| Virtual Tennis b | 0 / 749 | 9 / 8 | 1 / 1 | yes | OK |

Pipeline performance:

| Clip | Processed frames | Time | FPS | Notes |
|---|---:|---:|---:|---|
| Virtual Tennis a | 200 | 12.0 s | 16.6 | everything drawn + MP4 written |
| video_input2 | 150 | 7.2 s | 20.9 | real broadcast |
| video_input2 (full clip) | 929 | 53.7 s | 17.3 overall / 18.0 steady | every frame processed, 20 MB output |
| video_input5 | 120 | 5.5 s | 21.8 | clay court |
| video_input8 (full clip) | 168 | 9.7 s | 17.2 overall / 21.6 steady | every frame processed |
| MJPEG stream test | 30 | 1.4 s | 21.0 | `start_mjpeg_stream.py` workflow |

Annotated outputs were written to:

- `output_virtual_a.mp4`, `output_virtual_b.mp4`
- `output_video_input2.mp4`, `output_video_input2_full.mp4`,
  `output_video_input5.mp4`, `output_video_input8.mp4`
- `output_stream_test.mp4`, `output_sample.mp4`
- `output_keypoints_test.mp4` (Virtual Tennis, court overlay)
- `output_keypoints_broadcast.mp4` (Vienna broadcast, court overlay)

Court-keypoint overlay was verified visually: the yellow skeleton follows the
real lines, all 14 numbers sit on the correct intersections, and the status
line reads `court keypoints: OK` on hard, clay and virtual courts.

## 7. Court keypoint overlay (troubleshooting)

`main.py` draws the terrain/court detection by default:

- green numbered points 0–3: outer court corners/lines
- orange numbered points 4–7: singles sidelines
- magenta numbered points 8–13: service lines / centre service line
- yellow lines: inferred court skeleton
- bottom-left status: `court keypoints: OK` or `court keypoints: FIT FAILED`

Flags: `--no-keypoints`, `--no-court-lines`, `--no-court-status`.

A failed homography is now visible in two ways: the status line turns red, and
the minimap is skipped instead of raising. For a single-frame still:

```bash
python tools/verify_models.py --video clip.mp4 --frame 0 --save /tmp/kp.jpg
```

## 8. Portable export to full Linux (Omarchy / Arch)

The project is ready to export. Do **not** copy `models/*.trt` — TensorRT plans
are machine-specific and the old WSL/Windows plan already demonstrated the
`Platform specific tag mismatch` failure. The new setup script rebuilds them.

See [`OMARCHY_SETUP.md`](OMARCHY_SETUP.md) for:

- what to export / exclude,
- NVIDIA driver and `uv` setup on Omarchy,
- manual and one-command environment setup,
- model download/export on the target machine,
- troubleshooting,
- a ready-to-paste prompt for a new OpenCode session (plus `AGENTS.md`).

## 9. Known limitations / recommendations

- TensorRT engines are tied to the TensorRT version and GPU architecture. Rebuild
  with `python tools/setup_models.py --all --overwrite` when upgrading either.
- The detector is stock COCO RT-DETR: it returns all 80 classes, then code filters
  `person` (0) and `sports ball` (32). Detecting only players/balls with a
  fine-tuned model would reduce false positives and speed up post-processing.
- The keypoint model is broadcast/elevated-camera oriented; extremely low or
  heavily cropped cameras may produce a bad court fit. The minimap now skips
  drawing when the homography is invalid instead of crashing.
- On file input, the whole file is processed sequentially. On network streams,
  frames are skipped when processing cannot keep up (original design).
- For headless WSL use `--no-display`; `opencv-python-headless` has no GUI.
- If a new environment is created, install PyTorch CUDA **before**
  `requirements.txt`, and keep `tensorrt==10.8.0.43` for CUDA 12.
