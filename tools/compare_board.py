#!/usr/bin/env python3
"""Compare board axmodel outputs (axengine) with the fp32 ONNX reference."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    ref = np.load(args.cases, allow_pickle=False)
    brd = np.load(args.board, allow_pickle=False)

    groups = [
        ("chunk0", "ref0_", "board0_"),
        ("chunk1 (ONNX states)", "ref1_", "board1_"),
        ("chunk1 (board state carry)", "ref1_", "board1carry_"),
    ]
    rows = []
    for label, rp, bp in groups:
        for k in ref.files:
            if not k.startswith(rp):
                continue
            name = k[len(rp):]
            bk = bp + name
            if bk not in brd:
                continue
            a, b = ref[k], brd[bk]
            rows.append((label, name, tuple(a.shape), cos(a, b) if a.dtype.kind == "f" else 1.0,
                         float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))
                         if a.dtype.kind in "iu" else 0.0))
    for name, ref_key, brd_key in [
        ("decoder_out", "ref_dec_out", "board_dec_out"),
        ("logit", "ref_logit", "board_logit"),
    ]:
        rows.append(("decoder/joiner", name, tuple(ref[ref_key].shape),
                     cos(ref[ref_key], brd[brd_key]), 0.0))

    float_rows = [r for r in rows if r[4] == 0.0 or True]
    worst = min(r[3] for r in float_rows if r[3] is not None)
    bad_int = [r for r in rows if r[4] > 0]
    lines = ["# Board axmodel vs fp32 ONNX", "",
             f"- worst float cosine: **{worst:.8f}**",
             f"- int state max diff: {max([r[4] for r in rows] + [0])}",
             "", "| group | tensor | shape | cosine | int_max_diff |",
             "|---|---|---|---|---|"]
    for label, name, shape, c, di in rows:
        lines.append(f"| {label} | {name} | {shape} | {c:.8f} | {di:.0f} |")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"worst cosine = {worst:.8f}, int mismatches = {len(bad_int)}")
    print(f"report: {args.report}")
    if worst < 0.99:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
