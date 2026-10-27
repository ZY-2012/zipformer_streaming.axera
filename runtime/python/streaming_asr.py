#!/usr/bin/env python3
"""Board-side streaming zipformer (AX650 AXMODEL) runtime.

Reads 80-dim kaldi fbank features (.npy [T,80]) and runs the three AXMODELs
chunk by chunk (chunk=32, T=39), carrying all 35 encoder states.
Only numpy + axengine are required, so it runs directly on the board.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from axengine import InferenceSession


def load_tokens(path: str) -> Dict[int, str]:
    table: Dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            token, _, idx = line.rstrip("\n").rpartition(" ")
            if token:
                table[int(idx)] = token
    return table


def tokens_to_text(tokens: List[int], table: Dict[int, str]) -> str:
    return "".join(table.get(i, "") for i in tokens).replace("\u2581", " ").strip()


class StreamingZipformer:
    def __init__(self, model_dir: str, tokens: str, chunk: int = 32):
        self.chunk = chunk
        self.T = chunk + 7
        self.enc = InferenceSession(str(Path(model_dir) / "encoder.axmodel"))
        self.dec = InferenceSession(str(Path(model_dir) / "decoder.axmodel"))
        self.join = InferenceSession(str(Path(model_dir) / "joiner.axmodel"))
        self.tokens = load_tokens(tokens)

        self.enc_out_names = [o.name for o in self.enc.get_outputs()]
        self.state_names = [i.name for i in self.enc.get_inputs() if i.name != "x"]
        self.state_shapes = {i.name: list(i.shape) for i in self.enc.get_inputs()}
        self.dec_in_name = self.dec.get_inputs()[0].name
        self.dec_out_name = self.dec.get_outputs()[0].name
        self.join_in_names = [i.name for i in self.join.get_inputs()]
        self.join_out_name = self.join.get_outputs()[0].name
        self.blank = 0
        self.context = 2

    def init_states(self) -> Dict[str, np.ndarray]:
        states = {}
        for name in self.state_names:
            shape = self.state_shapes[name]
            dtype = np.int32 if name.startswith("cached_len") else np.float32
            states[name] = np.zeros(shape, dtype=dtype)
        return states

    def reset(self) -> None:
        self.states = self.init_states()
        self.hyp: List[int] = [self.blank] * self.context
        y = np.asarray([self.hyp[-self.context:]], dtype=np.int32)
        self.dec_out = self.dec.run([self.dec_out_name], {self.dec_in_name: y})[0]

    def run_encoder(self, x: np.ndarray) -> np.ndarray:
        feed = {"x": np.ascontiguousarray(x, dtype=np.float32)}
        feed.update(self.states)
        outs = self.enc.run(self.enc_out_names, feed)
        out = dict(zip(self.enc_out_names, outs))
        for name in self.state_names:
            self.states[name] = out["new_" + name]
        return out["encoder_out"]

    def decode_features(self, feats: np.ndarray) -> dict:
        self.reset()
        n = feats.shape[0]
        for start in range(0, n, self.chunk):
            seg = feats[start:start + self.T]
            if seg.shape[0] < self.T:
                seg = np.concatenate(
                    [seg, np.zeros((self.T - seg.shape[0], 80), dtype=np.float32)], axis=0)
            enc_out = self.run_encoder(seg[None, :, :])
            for t in range(enc_out.shape[1]):
                cur = enc_out[:, t:t + 1, :].reshape(1, -1).astype(np.float32)
                logits = self.join.run(
                    [self.join_out_name],
                    {self.join_in_names[0]: cur, self.join_in_names[1]: self.dec_out},
                )[0]
                sym = int(logits.reshape(-1).argmax())
                if sym != self.blank:
                    self.hyp.append(sym)
                    y = np.asarray([self.hyp[-self.context:]], dtype=np.int32)
                    self.dec_out = self.dec.run(
                        [self.dec_out_name], {self.dec_in_name: y})[0]
        ids = self.hyp[self.context:]
        return {"tokens": ids, "text": tokens_to_text(ids, self.tokens)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--features", required=True, help=".npy file or directory of .npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    asr = StreamingZipformer(args.model_dir, args.tokens, args.chunk)
    files = []
    p = Path(args.features)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.npy")))
    else:
        files = [str(p)]
    result = {}
    for f in files:
        feats = np.load(f).astype(np.float32)
        r = asr.decode_features(feats)
        result[os.path.basename(f)] = r
        print(f"{os.path.basename(f)}: {r['text']}")
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
