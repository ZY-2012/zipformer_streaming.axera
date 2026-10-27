#!/usr/bin/env python3
"""Host-side preparation for board validation:

1. compute kaldi fbank features for the test wavs and save as .npy (board
   runtime only needs numpy + axengine);
2. build one/two real streaming chunks with zero / carried states and save
   the ONNX reference outputs for per-model cosine checks on the board.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fbank(wav: torch.Tensor, sr: int) -> np.ndarray:
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    feats = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0), num_mel_bins=80, dither=0.0, snip_edges=False,
        sample_frequency=16000, high_freq=-400, frame_length=25,
        frame_shift=10, energy_floor=1e-10,
    )
    return feats.numpy().astype(np.float32)


def load_feats(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav[0]
    wav = torch.cat([wav, torch.zeros(int(0.3 * 16000))])
    return fbank(wav, sr)


def zero_states(sess) -> dict:
    out = {}
    for i in sess.get_inputs():
        if i.name == "x":
            continue
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        dtype = np.int64 if i.type == "tensor(int64)" else np.float32
        out[i.name] = np.zeros(shape, dtype=dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=str(PROJECT_ROOT / "build" / "onnx"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "build" / "board_validate"))
    ap.add_argument("--wavs", default=str(PROJECT_ROOT / "samples" / "*.wav"))
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out_dir)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for pattern in args.wavs.split(","):
        files.extend(sorted(glob.glob(pattern)))
    files = [f for f in files if f.lower().endswith((".wav", ".flac", ".mp3"))][:args.limit]
    for f in files:
        np.save(feat_dir / (Path(f).stem + ".npy"), load_feats(f))
    print(f"features: {len(files)} -> {feat_dir}")

    od = Path(args.onnx_dir)
    enc = ort.InferenceSession(str(od / "encoder-chunk32.sim.onnx"),
                               providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(od / "decoder-chunk32.sim.onnx"),
                               providers=["CPUExecutionProvider"])
    join = ort.InferenceSession(str(od / "joiner-chunk32.sim.onnx"),
                                providers=["CPUExecutionProvider"])
    enc_out_names = [o.name for o in enc.get_outputs()]

    feats = load_feats(files[0])
    if feats.shape[0] < 71:
        feats = np.concatenate([feats, np.zeros((71 - feats.shape[0], 80), np.float32)])
    x0 = feats[0:39][None]
    x1 = feats[32:71][None]
    state0 = zero_states(enc)
    out0 = dict(zip(enc_out_names, enc.run(None, {"x": x0, **state0})))
    state1 = {k: out0["new_" + k] for k in state0}
    out1 = dict(zip(enc_out_names, enc.run(None, {"x": x1, **state1})))

    rng = np.random.default_rng(20261027)
    y = rng.integers(0, 6254, size=(1, 2), dtype=np.int64)
    dec_out = dec.run(None, {"y": y})[0]
    frame = out0["encoder_out"][:, 0:1, :].reshape(1, -1)
    logit = join.run(None, {"encoder_out": frame, "decoder_out": dec_out})[0]

    arrays = {
        "x0": x0, "x1": x1, "y": y,
        "dec_out": dec_out, "logit": logit,
        "enc_frame0": frame,
    }
    for k, v in state0.items():
        arrays["s0_" + k] = v
    for k, v in state1.items():
        arrays["s1_" + k] = v
    for k, v in out0.items():
        arrays["ref0_" + k] = v
    for k, v in out1.items():
        arrays["ref1_" + k] = v
    arrays["ref_dec_out"] = dec_out
    arrays["ref_logit"] = logit
    np.savez(out / "cases.npz", **arrays)
    meta = {
        "feature_files": [str(feat_dir / (Path(f).stem + ".npy")) for f in files],
        "case_source": files[0],
        "state_names": list(state0),
        "enc_out_names": enc_out_names,
    }
    (out / "cases.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cases: {out / 'cases.npz'}")


if __name__ == "__main__":
    main()
