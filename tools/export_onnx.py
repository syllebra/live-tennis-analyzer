#!/usr/bin/env python3
"""Export an RT-DETRv2 checkpoint to ONNX.

This replaces the accidentally committed HTML page that used to live at this
path.  It is a thin wrapper around the logic in ``tools/setup_models.py``.

Examples
--------
    python tools/export_onnx.py \
        -c ../RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
        -r ../RT-DETR/models/rtdetrv2_r50vd_6x_coco_ema.pth \
        -o models/rtdetrv2_r50vd_6x_coco_ema.onnx --check

If ``-c`` is omitted, the r50 COCO config from the RT-DETR clone is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.setup_models import (  # noqa: E402
    DEFAULT_RTDETR_REPO,
    check_onnx,
    ensure_rtdetr_repo,
    export_rtdetr_onnx,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="RT-DETR YAML config")
    parser.add_argument("--resume", "-r", required=True, help="PyTorch checkpoint (.pth)")
    parser.add_argument("--output_file", "-o", default="models/model.onnx", help="output ONNX path")
    parser.add_argument("--input_size", "-s", type=int, default=640)
    parser.add_argument("--rtdetr-repo", default=str(DEFAULT_RTDETR_REPO))
    parser.add_argument("--check", action="store_true", help="run the ONNX checker")
    args = parser.parse_args()

    repo = ensure_rtdetr_repo(Path(args.rtdetr_repo).expanduser().resolve())
    config = (
        Path(args.config).expanduser().resolve()
        if args.config
        else repo / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_6x_coco.yml"
    )
    weights = Path(args.resume).expanduser().resolve()
    output = Path(args.output_file).expanduser().resolve()
    if not config.exists():
        parser.error(f"config not found: {config}")
    if not weights.exists():
        parser.error(f"weights not found: {weights}")

    export_rtdetr_onnx(repo, weights, config, output, input_size=args.input_size)
    if args.check:
        check_onnx(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
