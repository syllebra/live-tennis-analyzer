#!/usr/bin/env python3
"""Download and convert the models used by the live tennis analyzer.

The script produces:

  models/<rtdetr>.pth                 PyTorch RT-DETRv2 weights
  models/<rtdetr>.onnx                ONNX detector (images + orig_target_sizes)
  models/<rtdetr>.trt                 TensorRT FP16 engine
  models/keypoint_detector.pth        PyTorch keypoint weights (repo layout)
  models/keypoint_detector.onnx       ONNX keypoint model (1x3x224x224 -> 1x28)
  models/keypoint_detector.trt        TensorRT FP16 engine

Everything is done with Python APIs, so no `trtexec` binary is required.
Only the PyTorch/ONNX/TensorRT Python packages are needed.

Examples
--------
    # Everything, default RT-DETRv2-L (r50)
    python tools/setup_models.py --all

    # Smaller / faster detector
    python tools/setup_models.py --all --rtdetr-model r18

    # Only download weights, do not export anything
    python tools/setup_models.py --download-only

    # Rebuild engines from existing ONNX files
    python tools/setup_models.py --component keypoint --skip-onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RTDETR_REPO = PROJECT_ROOT.parent / "RT-DETR"

RTDETR_MODELS: Dict[str, Dict[str, str]] = {
    "r18": {
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "weights_name": "rtdetrv2_r18vd_120e_coco_ema.pth",
        "onnx_name": "rtdetrv2_r18vd_120e_coco_ema.onnx",
        "trt_name": "rtdetrv2_r18vd_120e_coco_ema.trt",
        "size_mb": "~78 MB",
    },
    "r34": {
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",
        "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth",
        "weights_name": "rtdetrv2_r34vd_120e_coco_ema.pth",
        "onnx_name": "rtdetrv2_r34vd_120e_coco_ema.onnx",
        "trt_name": "rtdetrv2_r34vd_120e_coco_ema.trt",
        "size_mb": "~62 MB",
    },
    "r50": {
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
        "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
        "weights_name": "rtdetrv2_r50vd_6x_coco_ema.pth",
        "onnx_name": "rtdetrv2_r50vd_6x_coco_ema.onnx",
        "trt_name": "rtdetrv2_r50vd_6x_coco_ema.trt",
        "size_mb": "~165 MB",
    },
    "r101": {
        "config": "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
        "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
        "weights_name": "rtdetrv2_r101vd_6x_coco_from_paddle.pth",
        "onnx_name": "rtdetrv2_r101vd_6x_coco_from_paddle.onnx",
        "trt_name": "rtdetrv2_r101vd_6x_coco_from_paddle.trt",
        "size_mb": "~294 MB",
    },
}

KEYPOINT_HF_URL = (
    "https://huggingface.co/Coddieharsh/tennis-court-keypoints/"
    "resolve/main/keypoints_model_geoaug.pth?download=true"
)
KEYPOINT_NAME = "keypoint_detector.pth"
KEYPOINT_CONVERTED_NAME = "keypoint_detector.pth"
KEYPOINT_ONNX_NAME = "keypoint_detector.onnx"
KEYPOINT_TRT_NAME = "keypoint_detector.trt"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[setup] WARNING: {msg}", file=sys.stderr, flush=True)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_html(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        head = handle.read(1024).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:256]


def valid_weight_file(path: Path, min_size: int = 1_000_000) -> bool:
    """Reject HTML error pages and truncated downloads."""
    if not path.exists() or path.stat().st_size < min_size:
        return False
    if looks_like_html(path):
        return False
    # PyTorch checkpoints are zip archives; ONNX starts with a protobuf-like
    # byte sequence but is never HTML.  The HTML check above is enough for ONNX.
    if path.suffix == ".pth":
        with path.open("rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    return True


def download_file(url: str, dest: Path, min_size: int = 1_000_000, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite and valid_weight_file(dest, min_size):
        log(f"using existing {dest} ({human_size(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    log(f"downloading {url}")
    log(f"         -> {dest}")
    request = urllib.request.Request(url, headers={"User-Agent": "live-tennis-analyzer-setup/1.0"})
    started = time.time()
    last_report = started
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_report >= 2.0:
                    if total:
                        pct = 100.0 * done / total
                        log(f"         {human_size(done)} / {human_size(total)} ({pct:5.1f}%)")
                    else:
                        log(f"         {human_size(done)}")
                    last_report = now
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    if not valid_weight_file(tmp, min_size):
        size = tmp.stat().st_size if tmp.exists() else 0
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded file is not a valid checkpoint ({size} bytes): {url}\n"
            "This usually means the server returned an HTML error/login page."
        )

    tmp.replace(dest)
    log(f"         done in {time.time() - started:.1f}s ({human_size(dest.stat().st_size)})")
    return dest


def ensure_rtdetr_repo(repo: Path) -> Path:
    if (repo / "rtdetrv2_pytorch" / "src" / "core" / "__init__.py").exists():
        return repo
    if repo.exists() and any(repo.iterdir()):
        raise RuntimeError(
            f"{repo} exists but does not look like the RT-DETR repository.\n"
            "Point --rtdetr-repo at a clone of https://github.com/lyuwenyu/RT-DETR"
        )
    log(f"cloning RT-DETR into {repo}")
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/lyuwenyu/RT-DETR.git", str(repo)],
        check=True,
    )
    return repo


# ---------------------------------------------------------------------------
# PyTorch -> ONNX
# ---------------------------------------------------------------------------


def export_rtdetr_onnx(
    repo: Path,
    weights: Path,
    config: Path,
    onnx_path: Path,
    input_size: int = 640,
) -> Path:
    import torch
    import torch.nn as nn

    pytorch_root = repo / "rtdetrv2_pytorch"
    if str(pytorch_root) not in sys.path:
        sys.path.insert(0, str(pytorch_root))

    from src.core import YAMLConfig  # noqa: WPS433

    log(f"loading RT-DETR config {config}")
    cfg = YAMLConfig(str(config), resume=str(weights))

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    # Training checkpoints may be wrapped by DataParallel/DDP.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().eval()
    dummy_images = torch.rand(1, 3, input_size, input_size)
    dummy_sizes = torch.tensor([[input_size, input_size]], dtype=torch.int64)

    with torch.no_grad():
        outputs = model(dummy_images, dummy_sizes)
    if isinstance(outputs, dict):
        shapes = ", ".join(f"{k}={tuple(v.shape)}" for k, v in outputs.items())
    else:
        # rtdetrv2 deploy() returns (labels, boxes, scores)
        shapes = ", ".join(
            f"{name}={tuple(value.shape)}"
            for name, value in zip(("labels", "boxes", "scores"), outputs)
        )
    log(f"RT-DETR forward pass OK: {shapes}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_images, dummy_sizes),
        str(onnx_path),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={
            "images": {0: "N"},
            "orig_target_sizes": {0: "N"},
            # Keep the 300 detections fixed; only the batch is dynamic.
        },
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )
    _inline_onnx_external_data(onnx_path)
    return onnx_path


def _remap_resnet50_state_to_repo_layout(state: Dict[str, "object"]) -> Dict[str, "object"]:
    """Map torchvision ResNet-50 keys to training/keypoint_training.KeypointModel keys.

    The Hugging Face checkpoint is a plain ``resnet50`` with ``fc`` replaced by
    ``Linear(2048, 28)``.  The repository wraps the children except ``fc`` in a
    ``Sequential`` called ``backbone`` and uses ``keypoint_head`` for the head.
    """
    import torchvision

    children = list(torchvision.models.resnet50(weights=None).named_children())
    child_index = {name: idx for idx, (name, _) in enumerate(children)}

    remapped = {}
    for key, value in state.items():
        head, sep, rest = key.partition(".")
        if head == "fc":
            remapped[f"keypoint_head.0.{rest}"] = value
        elif head in child_index:
            remapped[f"backbone.{child_index[head]}.{rest}"] = value
        elif key.startswith("keypoint_head.") or key.startswith("backbone."):
            remapped[key] = value
        else:
            # Unknown key; keep it so load_state_dict gives a useful error.
            remapped[key] = value
    return remapped


def _build_keypoint_model(backbone, num_keypoints: int = 14):
    """Same architecture as training/keypoint_training.KeypointModel.

    Kept local so the setup script does not need tensorboard/opencv just to
    export the model.
    """
    import torch.nn as nn

    class KeypointModel(nn.Module):
        def __init__(self, resnet_backbone, n_keypoints: int):
            super().__init__()
            output_size = list(resnet_backbone.children())[-1].in_features
            self.backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
            self.keypoint_head = nn.Sequential(nn.Linear(output_size, n_keypoints * 2))

        def forward(self, x):
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            return self.keypoint_head(features)

    return KeypointModel(backbone, num_keypoints)


def export_keypoint_onnx(weights: Path, onnx_path: Path, pth_path: Path) -> Path:
    import torch
    import torchvision

    log(f"loading keypoint weights {weights}")
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint and not any(
        k.startswith(("conv1.", "backbone.")) for k in checkpoint
    ):
        checkpoint = checkpoint["model"]
    state = {k.removeprefix("module."): v for k, v in checkpoint.items()}

    if any(k.startswith("keypoint_head.") for k in state):
        model_state = state
    elif "fc.weight" in state:
        model_state = _remap_resnet50_state_to_repo_layout(state)
    else:
        raise RuntimeError(
            "unrecognised keypoint checkpoint layout; first keys: "
            + ", ".join(list(state)[:8])
        )

    backbone = torchvision.models.resnet50(weights=None)
    model = _build_keypoint_model(backbone, num_keypoints=14)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # Save a checkpoint in the layout expected by training/keypoint_to_onnx.py.
    pth_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pth_path)
    log(f"saved PyTorch keypoint checkpoint -> {pth_path}")

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    log(f"keypoint forward pass OK: output={tuple(out.shape)}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )
    _inline_onnx_external_data(onnx_path)
    return onnx_path


def _inline_onnx_external_data(onnx_path: Path) -> None:
    """Turn ``model.onnx`` + ``model.onnx.data`` into a single ONNX file.

    Some PyTorch versions write large initializers to a sidecar file even when
    ``external_data=False`` is requested.  TensorRT's ONNX parser is happiest
    with a self-contained file, so inline the tensors if necessary.
    """
    sidecar = Path(str(onnx_path) + ".data")
    if not sidecar.exists():
        return
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            f"{sidecar} exists but the onnx package is not installed; cannot inline external data"
        ) from exc
    log(f"inlining external ONNX data from {sidecar.name}")
    model = onnx.load(str(onnx_path))  # load_external_data=True by default
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".inline")
    onnx.save_model(model, str(tmp), save_as_external_data=False)
    tmp.replace(onnx_path)
    sidecar.unlink(missing_ok=True)


def check_onnx(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        warn("onnx package not installed; skipping ONNX checker")
        return
    if looks_like_html(path):
        raise RuntimeError(f"{path} is an HTML page, not an ONNX model")
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    inputs = ", ".join(f"{i.name}{[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}" for i in model.graph.input)
    outputs = ", ".join(f"{o.name}{[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]}" for o in model.graph.output)
    log(f"ONNX check OK: {path.name} ({human_size(path.stat().st_size)})")
    log(f"         inputs:  {inputs}")
    log(f"         outputs: {outputs}")


# ---------------------------------------------------------------------------
# ONNX -> TensorRT
# ---------------------------------------------------------------------------


def build_trt_engine(
    onnx_path: Path,
    trt_path: Path,
    fp16: bool = True,
    workspace_gb: int = 4,
    batch_size: int = 1,
) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not installed.  Install it with:\n"
            "  pip install tensorrt==10.13.3.9\n"
            "or use --skip-trt to only create ONNX files."
        ) from exc

    log(f"TensorRT version: {trt.__version__}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    log(f"parsing {onnx_path.name}")
    if not parser.parse_from_file(str(onnx_path)):
        messages = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{messages}")

    # TensorRT 10.8 exposes create_builder_config(); newer versions prefer
    # create_configuration(network).
    if hasattr(builder, "create_configuration"):
        config = builder.create_configuration(network)
    else:
        config = builder.create_builder_config()
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log("FP16 enabled")
        else:
            warn("platform does not report fast FP16; building FP32 engine")
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)

    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = []
        for dim_index, dim in enumerate(tensor.shape):
            if dim < 0:
                # Default to batch_size for a leading dynamic dimension and a
                # value of 2 for orig_target_sizes-style [N, 2] inputs.
                if dim_index == 0:
                    shape.append(batch_size)
                else:
                    shape.append(2)
            else:
                shape.append(dim)
        shape = tuple(shape)
        profile.set_shape(tensor.name, shape, shape, shape)
        log(f"optimization profile: {tensor.name} {shape} (min=opt=max)")
    config.add_optimization_profile(profile)

    log("building TensorRT engine (this can take several minutes)...")
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see log above)")

    trt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = trt_path.with_suffix(trt_path.suffix + ".part")
    with tmp.open("wb") as handle:
        handle.write(serialized)
    tmp.replace(trt_path)
    log(f"engine written -> {trt_path} ({human_size(trt_path.stat().st_size)}) in {time.time() - started:.1f}s")
    return trt_path


def validate_trt_engine(trt_path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with trt_path.open("rb") as handle:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {trt_path}")
    io = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        io.append(f"{'in' if mode == trt.TensorIOMode.INPUT else 'out'}:{name}{tuple(engine.get_tensor_shape(name))}")
    log(f"engine OK: {trt_path.name} -> " + ", ".join(io))


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def resolve_rtdetr_inputs(repo: Path, key: str, models_dir: Path, overwrite: bool) -> Tuple[Path, Path]:
    info = RTDETR_MODELS[key]
    repo = ensure_rtdetr_repo(repo)
    config = repo / "rtdetrv2_pytorch" / info["config"]
    if not config.exists():
        raise RuntimeError(f"RT-DETR config not found: {config}")

    weights = models_dir / info["weights_name"]
    if not (weights.exists() and valid_weight_file(weights)):
        # Try the already-downloaded copy in ../RT-DETR/models first.
        candidate = repo / "models" / info["weights_name"]
        if candidate.exists() and valid_weight_file(candidate):
            log(f"using existing {candidate}")
            shutil.copy2(candidate, weights)
        else:
            download_file(info["weights_url"], weights, min_size=10_000_000, overwrite=overwrite)

    # A local config copy is handy for reproducing the export, but the
    # authoritative config with all relative includes stays in the RT-DETR repo.
    local_config = models_dir / config.name
    if not local_config.exists() or overwrite:
        shutil.copy2(config, local_config)
    return weights, config


def run(args: argparse.Namespace) -> int:
    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(args.rtdetr_repo).expanduser().resolve()

    components = [args.component] if args.component != "all" else ["rtdetr", "keypoint"]
    manifest: Dict[str, Dict[str, object]] = {}

    if "rtdetr" in components:
        info = RTDETR_MODELS[args.rtdetr_model]
        log(f"=== RT-DETR ({args.rtdetr_model}, {info['size_mb']}) ===")
        weights, config = resolve_rtdetr_inputs(repo, args.rtdetr_model, models_dir, args.overwrite)
        onnx_path = models_dir / info["onnx_name"]
        trt_path = models_dir / info["trt_name"]

        if not args.download_only and not args.skip_onnx:
            if onnx_path.exists() and not args.overwrite:
                log(f"using existing {onnx_path}")
            else:
                export_rtdetr_onnx(repo, weights, config, onnx_path, input_size=args.input_size)
            check_onnx(onnx_path)

        if not args.download_only and not args.skip_trt:
            if trt_path.exists() and not args.overwrite:
                log(f"using existing {trt_path}")
            else:
                build_trt_engine(
                    onnx_path,
                    trt_path,
                    fp16=not args.fp32,
                    workspace_gb=args.workspace_gb,
                    batch_size=args.batch_size,
                )
            validate_trt_engine(trt_path)

        manifest["rtdetr"] = {
            "model": args.rtdetr_model,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "config": str(config),
            "onnx": str(onnx_path) if onnx_path.exists() else None,
            "trt": str(trt_path) if trt_path.exists() else None,
        }

    if "keypoint" in components:
        log("=== Tennis court keypoint detector ===")
        keypoint_weights = models_dir / KEYPOINT_NAME
        if args.keypoint_weights:
            source = Path(args.keypoint_weights).expanduser().resolve()
            if not source.exists():
                raise RuntimeError(f"--keypoint-weights not found: {source}")
            log(f"using local keypoint weights {source}")
            shutil.copy2(source, keypoint_weights)
        else:
            download_file(KEYPOINT_HF_URL, keypoint_weights, min_size=10_000_000, overwrite=args.overwrite)

        onnx_path = models_dir / KEYPOINT_ONNX_NAME
        trt_path = models_dir / KEYPOINT_TRT_NAME
        if not args.download_only and not args.skip_onnx:
            if onnx_path.exists() and not args.overwrite:
                log(f"using existing {onnx_path}")
            else:
                export_keypoint_onnx(keypoint_weights, onnx_path, keypoint_weights)
            check_onnx(onnx_path)

        if not args.download_only and not args.skip_trt:
            if trt_path.exists() and not args.overwrite:
                log(f"using existing {trt_path}")
            else:
                build_trt_engine(
                    onnx_path,
                    trt_path,
                    fp16=not args.fp32,
                    workspace_gb=args.workspace_gb,
                    batch_size=1,
                )
            validate_trt_engine(trt_path)

        manifest["keypoint"] = {
            "weights": str(keypoint_weights),
            "weights_sha256": sha256_file(keypoint_weights),
            "onnx": str(onnx_path) if onnx_path.exists() else None,
            "trt": str(trt_path) if trt_path.exists() else None,
        }

    manifest["environment"] = {
        "python": sys.version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for module_name in ("torch", "torchvision", "onnx", "tensorrt"):
        try:
            module = __import__(module_name)
            manifest["environment"][module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            manifest["environment"][module_name] = "not installed"

    manifest_path = models_dir / "models.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"manifest written -> {manifest_path}")
    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download tennis analyzer models and export ONNX/TensorRT engines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="output directory")
    parser.add_argument("--rtdetr-repo", default=str(DEFAULT_RTDETR_REPO), help="RT-DETR git clone")
    parser.add_argument("--rtdetr-model", choices=sorted(RTDETR_MODELS), default="r50", help="RT-DETRv2 variant")
    parser.add_argument("--component", choices=["all", "rtdetr", "keypoint"], default="all", help="which model(s) to prepare")
    parser.add_argument("--all", action="store_true", help="prepare both models (same as --component all)")
    parser.add_argument("--keypoint-weights", default=None, help="use a local keypoint checkpoint instead of downloading")
    parser.add_argument("--input-size", type=int, default=640, help="RT-DETR square input size")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size baked into the TensorRT profile")
    parser.add_argument("--workspace-gb", type=int, default=4, help="TensorRT workspace memory pool")
    parser.add_argument("--fp32", action="store_true", help="build FP32 engines instead of FP16")
    parser.add_argument("--download-only", action="store_true", help="download weights, do not export")
    parser.add_argument("--skip-onnx", action="store_true", help="reuse existing ONNX files")
    parser.add_argument("--skip-trt", action="store_true", help="do not build TensorRT engines")
    parser.add_argument("--overwrite", action="store_true", help="redownload/reexport existing files")
    args = parser.parse_args(argv)
    if args.all:
        args.component = "all"
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
