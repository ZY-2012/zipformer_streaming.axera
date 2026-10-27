#!/usr/bin/env python3
"""Streaming zipformer (zh-en) board runtime for AX650 AXMODELs.

Two input modes:
  --wav FILE|DIR      read 16 kHz mono PCM WAV, compute 80-dim kaldi fbank
                      (kaldi-native-fbank, same options as the C++ runtime)
  --features FILE|DIR read precomputed fbank features (.npy, shape [T,80])

Runs encoder/decoder/joiner chunk by chunk (chunk=32, encoder input T=39) and
carries the 35 encoder states.  Requires numpy + axengine; the WAV mode also
needs kaldi-native-fbank.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import wave
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


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != 16000 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path}: need 16 kHz mono 16-bit PCM WAV "
                f"(channels={w.getnchannels()}, rate={w.getframerate()}, "
                f"sampwidth={w.getsampwidth()})"
            )
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return np.concatenate([audio, np.zeros(int(0.3 * 16000), dtype=np.float32)])


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    try:
        import kaldi_native_fbank as knf
    except ImportError as exc:
        raise SystemExit(
            "WAV mode needs kaldi-native-fbank: pip install kaldi-native-fbank"
        ) from exc
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    opts.energy_floor = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, audio)
    fbank.input_finished()
    return np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])


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
        for start in range(0, max(n - self.T + 1, 0), self.chunk):
            seg = np.ascontiguousarray(feats[start:start + self.T], dtype=np.float32)
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


def collect(path: str, suffix: str) -> List[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(glob.glob(str(p / f"*{suffix}")))
    return [str(p)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="16 kHz mono PCM WAV file or directory")
    src.add_argument("--features", help=".npy file or directory (shape [T,80])")
    ap.add_argument("--out", default="", help="optional result JSON")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--stats", action="store_true", help="print load/inference timing")
    args = ap.parse_args()

    t0 = time.perf_counter()
    asr = StreamingZipformer(args.model_dir, args.tokens, args.chunk)
    load_seconds = time.perf_counter() - t0
    if args.wav:
        files = collect(args.wav, ".wav")
        audio = {os.path.basename(f): read_wav(f) for f in files}
        feats = {k: compute_fbank(v) for k, v in audio.items()}
    else:
        files = collect(args.features, ".npy")
        feats = {os.path.basename(f): np.load(f).astype(np.float32) for f in files}

    result = {}
    for name, f in feats.items():
        t1 = time.perf_counter()
        r = asr.decode_features(f)
        infer = time.perf_counter() - t1
        result[name] = r
        print(f"{name}: {r['text']}")
        if args.stats:
            msg = f"  load={load_seconds:.3f}s infer={infer:.3f}s"
            if args.wav:
                dur = len(audio[name]) / 16000.0
                msg += f" audio={dur:.2f}s rtf={infer / max(dur, 1e-9):.3f}"
            print(msg)
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
