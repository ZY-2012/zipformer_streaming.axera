#!/usr/bin/env python3
"""Create three isolated Magnetar task dirs (encoder/decoder/joiner), copy the
static ONNX, write model_meta.json and the Pulsar2 config used by
magnetar.stages.compile.run(custom_config=...)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import onnx


DTYPES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
}


def vi_shape(vi) -> list[int]:
    return [d.dim_value for d in vi.type.tensor_type.shape.dim]


def vi_dtype(vi) -> str:
    return DTYPES.get(vi.type.tensor_type.elem_type, str(vi.type.tensor_type.elem_type))


def build_meta(model_name: str, onnx_path: Path) -> dict:
    m = onnx.load(str(onnx_path))
    inits = {i.name for i in m.graph.initializer}
    inputs = [
        {"name": vi.name, "shape": vi_shape(vi), "dtype": vi_dtype(vi), "layout": "NCHW"}
        for vi in m.graph.input if vi.name not in inits
    ]
    outputs = [
        {"name": vi.name, "shape": vi_shape(vi), "dtype": vi_dtype(vi)}
        for vi in m.graph.output
    ]
    return {
        "model_name": model_name,
        "framework": "pytorch",
        "inputs": inputs,
        "outputs": outputs,
        "opset": 18,
        "torch_onnx_cosine": 1.0,
        "source": "icefall pruned_transducer_stateless7_streaming, chunk=32",
    }


def build_config(onnx_path: Path, calib_dir_in_container: str = "/workspace/export/calib_data") -> dict:
    m = onnx.load(str(onnx_path))
    inits = {i.name for i in m.graph.initializer}
    inputs = [vi for vi in m.graph.input if vi.name not in inits]
    input_shapes = ",".join(
        f"{vi.name}:" + "x".join(str(d.dim_value) for d in vi.type.tensor_type.shape.dim)
        for vi in inputs
    )
    input_configs = [
        {
            "tensor_name": vi.name,
            "calibration_dataset": f"{calib_dir_in_container}/{vi.name}.zip",
            "calibration_size": -1,
            "calibration_format": "Numpy",
        }
        for vi in inputs
    ]
    return {
        "model_type": "ONNX",
        "npu_mode": "NPU3",
        "target_hardware": "AX650",
        "input": "/workspace/export/model.onnx",
        "output_name": "model.axmodel",
        "output_dir": "/workspace/compile",
        "work_dir": "/workspace/compile/work",
        "input_shapes": input_shapes,
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": True,
            "model_check": True,
        },
        "quant": {
            "input_configs": input_configs,
            "layer_configs": [
                {
                    "start_tensor_names": ["DEFAULT"],
                    "end_tensor_names": ["DEFAULT"],
                    "data_type": "U16",
                },
                {"op_type": "Pow", "data_type": "U8"},
            ],
            "calibration_method": "MinMax",
            "enable_smooth_quant": True,
            "conv_bias_data_type": "FP32",
            "disable_auto_refine_scale": True,
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            "transformer_opt_level": 0,
            "highest_mix_precision": False,
        },
        "input_processors": [{"tensor_name": "DEFAULT"}],
        "compiler": {"check": 0},
    }


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/shared/huyuan/ASR/ZF/streaming")
    ap.add_argument("--chunk-tag", default="chunk32")
    args = ap.parse_args()
    root = Path(args.root)
    onnx_dir = root / "export" / "onnx"
    calib_dir = root / "calib" / "datasets"
    tasks_root = root / "magnetar_tasks"
    variants = {
        "encoder": onnx_dir / f"encoder-{args.chunk_tag}.sim.onnx",
        "decoder": onnx_dir / f"decoder-{args.chunk_tag}.sim.onnx",
        "joiner": onnx_dir / f"joiner-{args.chunk_tag}.sim.onnx",
    }

    for name, src in variants.items():
        task = tasks_root / name
        for d in ["origin", "export", "compile", "simulate", "sdk/python",
                  "sdk/cpp", "runonboard", "package", "cache"]:
            (task / d).mkdir(parents=True, exist_ok=True)
        dst = task / "export" / "model.onnx"
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        meta = build_meta(f"zipformer_streaming_{name}", dst)
        (task / "export" / "model_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        cfg = build_config(dst)
        (task / "compile").mkdir(exist_ok=True)
        (task / "compile" / "custom_config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8")
        # calibration tensors, hardlinked to avoid duplicating ~1GB
        cdir = task / "export" / "calib_data"
        cdir.mkdir(exist_ok=True)
        missing = []
        for item in meta["inputs"]:
            z = calib_dir / f"{item['name']}.zip"
            if z.exists():
                link_or_copy(z, cdir / z.name)
            else:
                missing.append(z.name)
        (task / "config.json").write_text(json.dumps({
            "MODEL_NAME": f"zipformer_streaming_{name}",
            "TARGET_HARDWARE": "AX650",
            "SOURCE": str(src),
            "TASK_DIR": str(task),
            "SDK_LANG": "both",
        }, indent=2), encoding="utf-8")
        (task / "task.md").write_text(
            f"# streaming zipformer {name}\n\n- source: {src}\n"
            f"- onnx: export/model.onnx\n- chunk: 32 (T=39)\n"
            f"- missing calib: {missing}\n", encoding="utf-8")
        print(f"{name}: inputs={len(meta['inputs'])} outputs={len(meta['outputs'])} "
              f"calib_missing={missing}")

    print("tasks:", tasks_root)


if __name__ == "__main__":
    main()
