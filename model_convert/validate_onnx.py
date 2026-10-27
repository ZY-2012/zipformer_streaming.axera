#!/usr/bin/env python3
"""Validate the re-exported static ONNX against the original pfluo TorchScript
traces (the ground truth for these weights)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

STATE_NAMES = (
    [f"cached_len_{i}" for i in range(5)]
    + [f"cached_avg_{i}" for i in range(5)]
    + [f"cached_key_{i}" for i in range(5)]
    + [f"cached_val_{i}" for i in range(5)]
    + [f"cached_val2_{i}" for i in range(5)]
    + [f"cached_conv1_{i}" for i in range(5)]
    + [f"cached_conv2_{i}" for i in range(5)]
)
NEW_NAMES = ["new_" + n for n in STATE_NAMES]


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def ort_sess(p: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def zero_states(s: ort.InferenceSession) -> dict:
    out = {}
    for i in s.get_inputs():
        if i.name == "x":
            continue
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        dtype = np.int64 if i.type == "tensor(int64)" else np.float32
        out[i.name] = np.zeros(shape, dtype=dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-dir", default=str(REPO / "origin"))
    ap.add_argument("--onnx-dir", default=str(REPO / "build" / "onnx"))
    ap.add_argument("--report", default=str(REPO / "build" / "reports" / "export_validate.md"))
    args = ap.parse_args()
    od = Path(args.onnx_dir)
    origin = Path(args.origin_dir)

    ck = torch.load(origin / "exp/pretrained.pt", map_location="cpu",
                    weights_only=False)["model"]
    dec = torch.jit.load(str(origin / "exp/decoder_jit_trace.pt"), map_location="cpu").eval()
    joi = torch.jit.load(str(origin / "exp/joiner_jit_trace.pt"), map_location="cpu").eval()
    enc = torch.jit.load(str(origin / "exp/encoder_jit_trace.pt"), map_location="cpu").eval()
    enc_proj_w = ck["joiner.encoder_proj.weight"].detach().numpy()
    enc_proj_b = ck["joiner.encoder_proj.bias"].detach().numpy()
    dec_proj_w = ck["joiner.decoder_proj.weight"].detach().numpy()
    dec_proj_b = ck["joiner.decoder_proj.bias"].detach().numpy()

    def project(x, w, b):
        return x.reshape(-1, w.shape[1]) @ w.T + b

    sess = {
        "encoder": ort_sess(od / "encoder-chunk32.sim.onnx"),
        "decoder": ort_sess(od / "decoder-chunk32.sim.onnx"),
        "joiner": ort_sess(od / "joiner-chunk32.sim.onnx"),
    }
    rows = []
    rng = np.random.default_rng(20261027)

    # decoder
    for _ in range(5):
        y = rng.integers(0, 6254, size=(1, 2), dtype=np.int64)
        with torch.no_grad():
            ref = dec(torch.from_numpy(y), torch.tensor(False)).reshape(1, -1).numpy()
        ref = project(ref, dec_proj_w, dec_proj_b)
        ours = sess["decoder"].run(None, {"y": y})[0]
        rows.append(("decoder_out", ours.shape, cos(ref, ours), 0.0))

    # encoder, including a second chunk with state carry
    states_t = None
    x_prev = None
    for trial in range(3):
        x = (rng.standard_normal((1, 39, 80)) * 0.5).astype(np.float32)
        feed = zero_states(sess["encoder"])
        feed["x"] = x
        if states_t is not None:
            for k, v in zip(STATE_NAMES, states_t):
                feed[k] = v.numpy()
        with torch.no_grad():
            ref_out, _, ref_states = enc(
                torch.from_numpy(x), torch.tensor([39]),
                [torch.from_numpy(feed[k]) for k in STATE_NAMES],
            )
        ref_proj = project(ref_out.numpy(), enc_proj_w, enc_proj_b)
        ours = dict(zip([o.name for o in sess["encoder"].get_outputs()],
                        sess["encoder"].run(None, feed)))
        rows.append(("encoder_out" + (f" (chunk{trial + 1})" if trial else ""),
                     ours["encoder_out"].shape, cos(ref_proj, ours["encoder_out"]), 0.0))
        for j, name in enumerate(NEW_NAMES):
            rows.append((name, ours[name].shape,
                         cos(ref_states[j].numpy(), ours[name]), 0.0))
        states_t = [ref_states[j] for j in range(len(ref_states))]

    # joiner: jit consumes raw 384/512, our ONNX consumes projected 512/512
    for _ in range(3):
        raw_e = (rng.standard_normal((1, 384)) * 0.5).astype(np.float32)
        raw_d = (rng.standard_normal((1, 512)) * 0.5).astype(np.float32)
        with torch.no_grad():
            ref = joi(torch.from_numpy(raw_e), torch.from_numpy(raw_d)).numpy()
        ours = sess["joiner"].run(None, {
            "encoder_out": project(raw_e, enc_proj_w, enc_proj_b).astype(np.float32),
            "decoder_out": project(raw_d, dec_proj_w, dec_proj_b).astype(np.float32),
        })[0]
        rows.append(("logit", ours.shape, cos(ref, ours), 0.0))

    worst = min(r[2] for r in rows)

    lines = [
        "# Streaming zipformer static ONNX validation",
        "",
        "## Ground truth: original pfluo TorchScript traces",
        "",
        f"- worst cosine: **{worst:.10f}** (threshold 0.999)",
        "- chunk=32, T=39, batch=1, opset=18",
        "",
        "| tensor | shape | cosine |",
        "|---|---|---|",
    ] + [f"| {n} | {tuple(s)} | {c:.10f} |" for n, s, c, _ in rows] + [
        "",
        "The `sherpa-onnx-...-2023-02-20` checkpoint is a different (older)",
        "pfluo export and is not used as a weight-level reference.",
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"worst cosine = {worst:.10f}")
    print(f"report: {report}")
    if worst < 0.999:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
