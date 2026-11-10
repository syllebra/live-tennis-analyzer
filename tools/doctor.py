#!/usr/bin/env python3
"""Environment and model health check.

Run this first in a new shell / new OpenCode session:

    source .venv/bin/activate
    python tools/doctor.py

Exit code 0 means the core runtime and both TensorRT engines are usable.
Exit code 1 means something is missing; the printed "Next steps" explain what.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {
    "object": PROJECT_ROOT / "models" / "rtdetrv2_r50vd_6x_coco_ema.trt",
    "keypoint": PROJECT_ROOT / "models" / "keypoint_detector.trt",
}


def module_version(name: str):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001 - report the real import error
        return None, exc


def run_command(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return None, exc


def find_nvidia_smi():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    wsl_path = Path("/usr/lib/wsl/lib/nvidia-smi")
    if wsl_path.exists():
        return str(wsl_path)
    return None


def check_engine(path: Path) -> tuple[bool, str]:
    try:
        import tensorrt as trt
    except Exception as exc:  # noqa: BLE001
        return False, f"tensorrt import failed: {exc}"

    if not path.exists():
        return False, "file missing"
    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with path.open("rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            return False, "deserialize_cuda_engine returned None"
        io = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            direction = "in" if mode == trt.TensorIOMode.INPUT else "out"
            io.append(f"{direction}:{name}{tuple(engine.get_tensor_shape(name))}")
        return True, ", ".join(io)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    object_engine = models_dir / DEFAULT_MODELS["object"].name
    keypoint_engine = models_dir / DEFAULT_MODELS["keypoint"].name

    report: dict = {
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "venv": os.environ.get("VIRTUAL_ENV"),
        "modules": {},
        "gpu": {},
        "models": {},
    }

    print("=" * 72)
    print("Tennis analyzer doctor")
    print("=" * 72)
    print(f"project:  {PROJECT_ROOT}")
    print(f"platform: {platform.platform()}")
    print(f"python:   {sys.version.split()[0]} ({sys.executable})")
    print(f"venv:     {os.environ.get('VIRTUAL_ENV', '(not activated)')}")

    print("\n-- Python packages " + "-" * 53)
    required_modules = ("torch", "torchvision", "cv2", "numpy", "onnx", "tensorrt", "scipy")
    missing_modules = []
    for name in required_modules:
        version = module_version(name)
        if isinstance(version, tuple):
            report["modules"][name] = None
            missing_modules.append(name)
            print(f"  {name:12s} MISSING ({version[1]})")
        else:
            report["modules"][name] = version
            print(f"  {name:12s} {version}")

    print("\n-- OpenCV / display " + "-" * 53)
    cv2_version = report["modules"].get("cv2")
    if cv2_version:
        try:
            import cv2

            build = cv2.getBuildInformation()
            gui_line = next(
                (line.strip() for line in build.splitlines() if line.strip().startswith("GUI:")),
                "GUI: unknown",
            )
            display = os.environ.get("DISPLAY")
            wayland = os.environ.get("WAYLAND_DISPLAY")
            report["opencv"] = {
                "version": cv2_version,
                "gui": gui_line,
                "display": display,
                "wayland_display": wayland,
            }
            print(f"  cv2 {cv2_version} ({gui_line})")
            print(f"  DISPLAY={display or '(unset)'} WAYLAND_DISPLAY={wayland or '(unset)'}")
            if gui_line.upper().endswith("NONE"):
                print("  GUI: this is a headless OpenCV build; cv2.imshow is unavailable.")
                print("       Install `opencv-python` (not headless) for the preview window.")
        except Exception as exc:  # noqa: BLE001
            print(f"  OpenCV check failed: {exc}")

    print("\n-- GPU " + "-" * 65)
    nvidia_smi = find_nvidia_smi()
    if not nvidia_smi:
        print("  nvidia-smi: NOT FOUND")
        report["gpu"]["nvidia_smi"] = False
    else:
        print(f"  nvidia-smi: {nvidia_smi}")
        result = run_command([
            nvidia_smi,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        if isinstance(result, tuple) or result.returncode != 0:
            print(f"  query failed: {result}")
        else:
            for line in result.stdout.strip().splitlines():
                print(f"  GPU: {line.strip()}")
            report["gpu"]["nvidia_smi"] = True
            report["gpu"]["query"] = result.stdout.strip()

    torch_version = report["modules"].get("torch")
    if torch_version:
        try:
            import torch

            available = torch.cuda.is_available()
            report["gpu"]["torch_cuda"] = available
            print(f"  torch CUDA available: {available}")
            if available:
                print(f"  torch device: {torch.cuda.get_device_name(0)}")
                print(f"  torch CUDA runtime: {torch.version.cuda}")
            else:
                print("  torch was imported but CUDA is not available.")
        except Exception as exc:  # noqa: BLE001
            print(f"  torch check failed: {exc}")
            available = False
    else:
        available = False

    print("\n-- Models " + "-" * 62)
    for label, path in (("object", object_engine), ("keypoint", keypoint_engine)):
        if not path.exists():
            print(f"  {label:9s} {path.name}: MISSING")
            report["models"][label] = {"path": str(path), "ok": False, "detail": "missing"}
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        ok, detail = check_engine(path)
        report["models"][label] = {"path": str(path), "ok": ok, "detail": detail}
        print(f"  {label:9s} {path.name}: {'OK' if ok else 'INVALID'} ({size_mb:.1f} MiB)")
        print(f"            {detail}")

    print("\n-- Next steps " + "-" * 57)
    ready = (
        not missing_modules
        and available
        and all(item.get("ok") for item in report["models"].values())
    )

    if missing_modules:
        print("  Install missing Python packages:")
        print("    uv pip install -r requirements.txt")
        print("    # if TensorRT is still missing:")
        print("    python -m pip install tensorrt==10.8.0.43")
    if not available:
        print("  NVIDIA/CUDA is not ready.")
        print("  On Omarchy/Arch, verify the driver first: nvidia-smi")
        print("  Typical packages: sudo pacman -S nvidia nvidia-utils linux-headers")
        print("  Then reboot and re-run this script.")
    if not object_engine.exists() or not keypoint_engine.exists():
        print("  Download/export models:")
        print("    python tools/setup_models.py --all")
    elif not all(item.get("ok") for item in report["models"].values()):
        print("  One or more engines are invalid for this machine/TensorRT.")
        print("  Rebuild them locally (TensorRT plans are not portable):")
        print("    python tools/setup_models.py --all --overwrite")
    if ready:
        print("  Everything looks ready. Smoke test:")
        print("    python tools/verify_models.py --video videos/sample_tennis.mp4 --save /tmp/preview.jpg")
        print("    python main.py --stream_url videos/sample_tennis.mp4 --no-display --max-frames 60")

    report["ready"] = ready
    if args.json:
        print("\n" + json.dumps(report, indent=2, default=str))
    print("\n" + ("READY" if ready else "NOT READY"))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
