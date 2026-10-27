#!/usr/bin/env python3
"""Streaming greedy decode with ONNX Runtime (reference path).

Processes a wav in decode-chunk-sized segments, carrying the 35 encoder states,
mirroring the C++ streaming runtime. Used to compare fp32 / official / board
outputs token by token.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torchaudio


def load_tokens(path: str) -> Dict[int, str]:
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            token, _, idx = line.rstrip("\n").rpartition(" ")
            if token:
                table[int(idx)] = token
    return table


def fbank(wav: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    return torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        num_mel_bins=80,
        dither=0.0,
        snip_edges=False,
        sample_frequency=16000,
        high_freq=-400,
        frame_length=25,
        frame_shift=10,
        energy_floor=1e-10,
    )


def init_states(sess: ort.InferenceSession) -> Dict[str, np.ndarray]:
    states = {}
    for inp in sess.get_inputs():
        if inp.name == "x":
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype = np.int64 if inp.type == "tensor(int64)" else np.float32
        states[inp.name] = np.zeros(shape, dtype=dtype)
    return states


def decode_file(path: str, cfg: dict, enc, dec, join, tokens: Dict[int, str]) -> dict:
    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav[0]
    wav = torch.cat([wav, torch.zeros(int(0.3 * 16000))])
    feats = fbank(wav, sr)
    n_frames = feats.shape[0]
    chunk = int(cfg["decode_chunk_len"])
    T = int(cfg["segment_frames"])
    blank = int(cfg["blank_id"])
    ctx = int(cfg["context_size"])

    states = init_states(enc)
    hyp: List[int] = [blank] * ctx
    y = np.asarray([hyp[-ctx:]], dtype=np.int64)
    dec_out = dec.run(None, {dec.get_inputs()[0].name: y})[0]

    enc_out_names = [o.name for o in enc.get_outputs()]
    for start in range(0, n_frames, chunk):
        seg = feats[start:start + T]
        if seg.shape[0] < T:
            seg = torch.cat([seg, torch.zeros(T - seg.shape[0], 80)], dim=0)
        x = seg.unsqueeze(0).numpy().astype(np.float32)
        feed = {"x": x, **states}
        outs = dict(zip(enc_out_names, enc.run(None, feed)))
        enc_out = outs["encoder_out"]
        for name in states:
            states[name] = outs["new_" + name]
        for t in range(enc_out.shape[1]):
            cur = enc_out[:, t:t + 1, :].reshape(1, -1)
            logits = join.run(None, {
                join.get_inputs()[0].name: cur,
                join.get_inputs()[1].name: dec_out,
            })[0]
            sym = int(logits.reshape(-1).argmax())
            if sym != blank:
                hyp.append(sym)
                y = np.asarray([hyp[-ctx:]], dtype=np.int64)
                dec_out = dec.run(None, {dec.get_inputs()[0].name: y})[0]

    ids = hyp[ctx:]
    text = "".join(tokens.get(i, "") for i in ids).replace("\u2581", " ").strip()
    return {"text": text, "tokens": ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--onnx-dir", default="")
    ap.add_argument("--encoder", default="")
    ap.add_argument("--decoder", default="")
    ap.add_argument("--joiner", default="")
    ap.add_argument("--wavs", required=True, help="glob(s), comma separated")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    od = Path(args.onnx_dir) if args.onnx_dir else cfg_path.parent
    enc_path = args.encoder or str(od / Path(cfg["onnx"]["encoder"]).name)
    dec_path = args.decoder or str(od / Path(cfg["onnx"]["decoder"]).name)
    join_path = args.joiner or str(od / Path(cfg["onnx"]["joiner"]).name)
    tokens = load_tokens(args.tokens or cfg["tokens"])

    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    join = ort.InferenceSession(join_path, providers=["CPUExecutionProvider"])

    files: List[str] = []
    for pattern in args.wavs.split(","):
        files.extend(sorted(glob.glob(pattern)))
    files = [f for f in files if f.lower().endswith((".wav", ".flac", ".mp3"))]
    if args.limit:
        files = files[:args.limit]

    result = {}
    for f in files:
        r = decode_file(f, cfg, enc, dec, join, tokens)
        result[os.path.basename(f)] = r
        print(f"{os.path.basename(f)}: {r['text']}")
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
