#!/usr/bin/env python3
"""Compare the re-exported static streaming ONNX with the official sherpa-onnx
streaming zipformer ONNX (same pfluo checkpoint)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

OFFICIAL = Path(
    "/data/shared/huyuan/ASR/ZF/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.HF"
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def sess(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def zero_inputs(s: ort.InferenceSession, names=None) -> dict:
    feed = {}
    for inp in s.get_inputs():
        if names is not None and inp.name not in names:
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype = np.int64 if inp.type == "tensor(int64)" else np.float32
        feed[inp.name] = np.zeros(shape, dtype=dtype)
    return feed


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray, rows: list) -> None:
    c = cosine(a, b)
    rows.append((name, a.shape, float(c), float(np.max(np.abs(a - b)))))
    if c < 0.999:
        print(f"  WARN {name}: cosine={c:.6f} shape_a={a.shape} shape_b={b.shape}")


def run_encoder(ours: Path, ref: Path, rows: list) -> None:
    so, sr = sess(ours), sess(ref)
    state_names = [i.name for i in so.get_inputs() if i.name != "x"]
    for trial in range(3):
        rng = np.random.default_rng(20261027 + trial)
        feed = zero_inputs(sr)
        x = (rng.standard_normal((1, 39, 80)) * 0.5).astype(np.float32)
        feed["x"] = x
        # our static model uses the exact same state tensors
        feed_ours = {k: v.copy() for k, v in feed.items() if k in {i.name for i in so.get_inputs()}}
        out_ref = dict(zip([o.name for o in sr.get_outputs()], sr.run(None, feed)))
        out_ours = dict(zip([o.name for o in so.get_outputs()], so.run(None, feed_ours)))
        for name in out_ref:
            if name not in out_ours:
                print(f"  MISSING output in ours: {name}")
                continue
            compare_arrays(name, out_ref[name], out_ours[name], rows)
        # second chunk with real previous states (checks state carry)
        if trial == 0:
            feed2 = zero_inputs(sr)
            feed2["x"] = (rng.standard_normal((1, 39, 80)) * 0.5).astype(np.float32)
            for name, val in out_ref.items():
                if name.startswith("new_"):
                    feed2[name[4:]] = val
            feed2_ours = {k: v.copy() for k, v in feed2.items()
                          if k in {i.name for i in so.get_inputs()}}
            out_ref2 = dict(zip([o.name for o in sr.get_outputs()], sr.run(None, feed2)))
            out_ours2 = dict(zip([o.name for o in so.get_outputs()], so.run(None, feed2_ours)))
            for name in out_ref2:
                compare_arrays(name + " (chunk2)", out_ref2[name], out_ours2[name], rows)


def run_decoder_joiner(ours_dir: Path, rows: list) -> None:
    so = sess(ours_dir / "decoder-chunk32.sim.onnx")
    sr = sess(OFFICIAL / "decoder-epoch-99-avg-1.onnx")
    rng = np.random.default_rng(7)
    for _ in range(3):
        y = rng.integers(0, 6254, size=(1, 2), dtype=np.int64)
        a = so.run(None, {"y": y})[0]
        b = sr.run(None, {"y": y})[0]
        compare_arrays("decoder_out", a, b, rows)

    so = sess(ours_dir / "joiner-chunk32.sim.onnx")
    sr = sess(OFFICIAL / "joiner-epoch-99-avg-1.onnx")
    for _ in range(3):
        e = rng.standard_normal((1, 512)).astype(np.float32)
        d = rng.standard_normal((1, 512)).astype(np.float32)
        a = so.run(None, {"encoder_out": e, "decoder_out": d})[0]
        b = sr.run(None, {"encoder_out": e, "decoder_out": d})[0]
        compare_arrays("logit", a, b, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=str(Path(__file__).resolve().parent / "onnx"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "reports" / "export_validate.md"))
    args = ap.parse_args()
    onnx_dir = Path(args.onnx_dir)

    rows: list = []
    print("== encoder ==")
    run_encoder(onnx_dir / "encoder-chunk32.sim.onnx",
                OFFICIAL / "encoder-epoch-99-avg-1.onnx", rows)
    print("== decoder / joiner ==")
    run_decoder_joiner(onnx_dir, rows)

    worst = min(r[2] for r in rows)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Static ONNX vs official sherpa-onnx validation", "",
             f"- worst cosine: **{worst:.8f}**", "",
             "| output | shape | cosine | max_abs_diff |", "|---|---|---|---|"]
    for name, shape, c, m in rows:
        lines.append(f"| {name} | {tuple(shape)} | {c:.8f} | {m:.3e} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"worst cosine = {worst:.8f}")
    print(f"report: {report}")
    if worst < 0.999:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
