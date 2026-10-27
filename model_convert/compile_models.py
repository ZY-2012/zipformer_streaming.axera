#!/usr/bin/env python3
"""Run Magnetar's COMPILE stage for the three zipformer submodels.

Magnetar's generic pipeline is single-input; we pass a custom Pulsar2 config
that lists every input tensor and its calibration archive. All temporary and
output files stay inside the task directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAGNETAR_ROOT = Path(os.environ.get("MAGNETAR_ROOT", PROJECT_ROOT / "Magnetar"))
if not (MAGNETAR_ROOT / "magnetar").is_dir():
    raise SystemExit(
        "Magnetar not found. Clone https://github.com/AXERA-TECH/Magnetar "
        f"next to this project or set MAGNETAR_ROOT (current: {MAGNETAR_ROOT})"
    )
sys.path.insert(0, str(MAGNETAR_ROOT))

from magnetar.stages import compile as mcompile  # noqa: E402
import magnetar.docker_util as _du  # noqa: E402

_ORIG_DOCKER = _du.docker_pulsar2


def _patched_docker(image, workspace, command, timeout=3600, **kwargs):
    return _ORIG_DOCKER(image, workspace, command, timeout=_TIMEOUT, **kwargs)


_du.docker_pulsar2 = _patched_docker
_TIMEOUT = 7200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", default=str(PROJECT_ROOT / "build" / "magnetar_tasks"))
    ap.add_argument("--image", default=os.environ.get(
        "PULSAR2_IMAGE", "docker-registry.aitsw.axera-tech.com/pulsar2:7.0"))
    ap.add_argument("--models", default="decoder,joiner,encoder")
    ap.add_argument("--precision", default="U16", choices=["U16", "U8"])
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    global _TIMEOUT
    _TIMEOUT = args.timeout
    root = Path(args.task_root)
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        task = root / name
        cfg = json.loads((task / "compile" / "custom_config.json").read_text(encoding="utf-8"))
        if args.precision == "U8":
            cfg["quant"]["layer_configs"] = [
                {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
                 "data_type": "U8"},
                {"op_type": "Pow", "data_type": "U8"},
            ]
        print(f"\n===== COMPILE {name} ({args.precision}) =====", flush=True)
        mcompile.run(task, "AX650", args.image, input_dtype="FP32",
                     custom_config=cfg, skip_validation=True)
        src = task / "compile" / "model.axmodel"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        dst = task / "compile" / f"{name}.axmodel"
        shutil.copy2(src, dst)
        summary = mcompile.summarize_compile_log(task)
        real_errors = [e for e in summary.get("errors", [])
                       if "it/s" not in e and "Analysing" not in e
                       and "Calibration Progress" not in e]
        summary["errors"] = real_errors
        print(f"[{name}] axmodel={src.stat().st_size / 1024 / 1024:.1f} MiB "
              f"macs={summary.get('macs')} errors={real_errors}", flush=True)
        (task / "compile" / "compile_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
