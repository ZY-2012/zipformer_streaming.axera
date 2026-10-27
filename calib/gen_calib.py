#!/usr/bin/env python3
"""生成 zipformer encoder/decoder/joiner 量化校准数据（torchaudio 版，无 kaldifeat 依赖）。

与官方 utils/save_inputs_data.py 逻辑对齐：
  - fbank: 80 维 kaldi 风格（dither=0, snip_edges=False, high_freq=-400, energy_floor=1e-10）
  - streaming chunk: chunk_len=96, x 为 103 帧（96 + 7 left context），不足补 0
  - cached 状态初始全 0 / cached_len=0，逐 chunk 用 encoder 输出的 new_cached_* 更新
  - greedy_search 驱动 decoder/joiner，收集真实输入
  - 输出 datasets/*.zip（pulsar2 Numpy calibration 格式，npy 带 batch 维）
"""
import argparse
import glob
import os
import zipfile

import numpy as np
import onnxruntime as ort
import torch
import torchaudio


def fbank_features(wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """kaldi 风格 fbank，与 kaldifeat OnlineFbank 配置对齐。返回 [T, 80]"""
    feats = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        num_mel_bins=80,
        dither=0.0,
        snip_edges=False,
        sample_frequency=sr,
        high_freq=-400,
        frame_length=25,
        frame_shift=10,
        energy_floor=1e-10,
    )
    return feats  # [T, 80]


class CalibCollector:
    def __init__(self, encoder_path, decoder_path, joiner_path, out_dir, max_wavs):
        self.out_dir = out_dir
        self.max_wavs = max_wavs
        os.makedirs(out_dir, exist_ok=True)
        self.enc = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
        self.join = ort.InferenceSession(joiner_path, providers=["CPUExecutionProvider"])

        self.enc_in = {i.name: i for i in self.enc.get_inputs()}
        self.enc_out_names = [o.name for o in self.enc.get_outputs()]
        self.context_size = 2
        self.blank_id = 0

        # cached 状态初始值（形状取自 encoder ONNX 输入）
        self.state_names = ["cached_len", "cached_avg", "cached_key", "cached_val",
                            "cached_val2", "cached_conv1", "cached_conv2"]
        self.states = {}
        for sname in self.state_names:
            names = sorted(n for n in self.enc_in if n.startswith(sname + "_"))
            arrs = []
            for n in names:
                t = self.enc_in[n]
                shape = [int(d) for d in t.shape]
                dtype = np.int64 if sname == "cached_len" else np.float32
                arrs.append(np.zeros(shape, dtype=dtype))
            self.states[sname] = arrs

        self.counter = 0
        self.pending = []  # (zip_path, npy_path, arcname)
        self.batch_size = 500

    # ---- zip 打包 ----
    def _add(self, zip_path, npy_path, arcname):
        self.pending.append((zip_path, npy_path, arcname))
        if len(self.pending) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        by_zip = {}
        for zp, np_, arc in self.pending:
            by_zip.setdefault(zp, []).append((np_, arc))
        for zp, items in by_zip.items():
            with zipfile.ZipFile(zp, "a", zipfile.ZIP_DEFLATED) as zf:
                for np_, arc in items:
                    zf.write(np_, arcname=arc)
        for _, np_, _ in self.pending:
            os.remove(np_)
        self.pending = []

    def _save_npy(self, zip_name, prefix, arr):
        self.counter += 1
        c = self.counter
        fname = f"{prefix}_{c}.npy"
        fpath = os.path.join(self.out_dir, fname)
        np.save(fpath, arr)
        self._add(os.path.join(self.out_dir, zip_name), fpath, fname)

    def save_encoder_inputs(self, x):
        self._save_npy("x.zip", "encoder_x", x.numpy())
        for sname, arrs in self.states.items():
            for i, s in enumerate(arrs):
                self._save_npy(f"{sname}_{i}.zip", f"encoder_{sname}_{i}", s)

    def run_encoder(self, x):
        self.save_encoder_inputs(x)
        feed = {"x": x.numpy()}
        for sname, arrs in self.states.items():
            for i, s in enumerate(arrs):
                feed[f"{sname}_{i}"] = s
        out = self.enc.run(self.enc_out_names, feed)
        encoder_out = torch.from_numpy(out[0])
        n_states = len(self.state_names)
        idx = 1
        for sname in self.state_names:
            for i in range(len(self.states[sname])):
                self.states[sname][i] = out[idx]
                idx += 1
        return encoder_out

    def run_decoder(self, decoder_input):
        self._save_npy("y.zip", "decoder_input", decoder_input.numpy())
        out = self.dec.run(None, {self.dec.get_inputs()[0].name: decoder_input.numpy()})[0]
        return torch.from_numpy(out)

    def run_joiner(self, encoder_out_t, decoder_out):
        self._save_npy("encoder_out.zip", "joiner_encoder_out", encoder_out_t.numpy())
        self._save_npy("decoder_out.zip", "joiner_decoder_out", decoder_out.numpy())
        out = self.join.run(
            None,
            {
                self.join.get_inputs()[0].name: encoder_out_t.numpy(),
                self.join.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]
        return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--joiner", required=True)
    ap.add_argument("--wavs-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-wavs", type=int, default=0, help="0 = all wavs")
    ap.add_argument("--chunk-len", type=int, default=96)
    ap.add_argument("--t", type=int, default=103, help="x 帧数 = chunk + 7")
    args = ap.parse_args()

    col = CalibCollector(args.encoder, args.decoder, args.joiner, args.output_dir, args.max_wavs)
    chunk, T = args.chunk_len, args.t
    wavs = sorted(glob.glob(os.path.join(args.wavs_dir, "*.wav")))
    if args.max_wavs:
        wavs = wavs[: args.max_wavs]
    print(f"wavs: {len(wavs)}")

    for wi, wf in enumerate(wavs):
        wave, sr = torchaudio.load(wf)
        if sr != 16000:
            wave = torchaudio.transforms.Resample(sr, 16000)(wave)
        wave = wave[0]
        wave = torch.cat([wave, torch.zeros(int(0.3 * 16000))])
        feats = fbank_features(wave)  # [Tt, 80]
        n_frames = feats.shape[0]

        hyp = [col.blank_id] * col.context_size
        decoder_input = torch.tensor([hyp], dtype=torch.int64)
        decoder_out = col.run_decoder(decoder_input)

        for start in range(0, n_frames, chunk):
            seg = feats[start : start + T]
            if seg.shape[0] < T:
                pad = torch.zeros(T - seg.shape[0], seg.shape[1])
                seg = torch.cat([seg, pad], dim=0)
            x = seg.unsqueeze(0)  # [1, T, 80]
            encoder_out = col.run_encoder(x)  # [1, T', 512]
            encoder_out = encoder_out.squeeze(0)
            for t in range(encoder_out.shape[0]):
                cur = encoder_out[t : t + 1]  # [1, 512]
                joiner_out = col.run_joiner(cur, decoder_out).squeeze(0)
                y = int(joiner_out.argmax(dim=0).item())
                if y != col.blank_id:
                    hyp.append(y)
                    decoder_input = torch.tensor([hyp[-col.context_size:]], dtype=torch.int64)
                    decoder_out = col.run_decoder(decoder_input)
        col._flush()
        print(f"[{wi + 1}/{len(wavs)}] {os.path.basename(wf)} done, samples={col.counter}")

    print("total input samples:", col.counter)
    for z in sorted(os.listdir(args.output_dir)):
        if z.endswith(".zip"):
            print(" ", z)


if __name__ == "__main__":
    main()
