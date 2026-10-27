#!/usr/bin/env python3
"""Export the streaming zipformer (pfluo/k2fsa-zipformer-chinese-english-mixed)
to fully static ONNX for AXERA Pulsar2.

Streaming recipe: pruned_transducer_stateless7_streaming
  - decode chunk = 32 fbank frames, encoder input T = chunk + 7 = 39
  - 5 encoder stages, layers 2,4,3,2,4, dims 384, attention 192
  - 35 streaming states (cached_len/avg/key/val/val2/conv1/conv2)
  - batch is fixed to 1 so every input/output dim is static

This script is intentionally k2/lhotse free: it only needs torch + onnx +
onnxsim and the vendored model code in export/model_code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model_code"))

from decoder import Decoder  # noqa: E402
from joiner import Joiner  # noqa: E402
from scaling_converter import convert_scaled_to_non_scaled  # noqa: E402
from zipformer import Zipformer  # noqa: E402

ENCODER_CFG = dict(
    num_encoder_layers=(2, 4, 3, 2, 4),
    feedforward_dims=(1024, 1024, 1536, 1536, 1024),
    nhead=(8, 8, 8, 8, 8),
    encoder_dims=(384, 384, 384, 384, 384),
    attention_dims=(192, 192, 192, 192, 192),
    encoder_unmasked_dims=(256, 256, 256, 256, 256),
    zipformer_downsampling_factors=(1, 2, 4, 8, 2),
    cnn_module_kernels=(31, 31, 31, 31, 31),
    num_left_chunks=4,
    short_chunk_size=50,
)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pt-model", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-len", type=int, default=32,
                   help="streaming decode chunk length in fbank frames")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--prefix", default="")
    p.add_argument("--no-cut-reciprocal", action="store_true")
    p.add_argument("--no-simplify", action="store_true")
    return p


def parse_tokens(path: str) -> Tuple[int, int, Dict[int, str]]:
    symbols: Dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            token, _, idx = line.rpartition(" ")
            symbols[int(idx)] = token
    blank_id = next(k for k, v in symbols.items() if v == "<blk>")
    # exclude graph disambiguation symbols (#1, #2, ...) from the model vocab
    vocab_size = 1 + max(k for k, v in symbols.items() if not v.startswith("#"))
    return blank_id, vocab_size, symbols


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_avg" in ckpt:
        sd = ckpt["model_avg"]
    else:
        sd = ckpt
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for pref in ("module.", "model."):
            if k.startswith(pref):
                k = k[len(pref):]
        out[k] = v
    return out


class Model(nn.Module):
    def __init__(self, encoder, decoder, joiner, encoder_dim: int,
                 decoder_dim: int, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        # only present in the training checkpoint; never exported
        self.simple_am_proj = nn.Linear(encoder_dim, vocab_size)
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)


class OnnxEncoder(nn.Module):
    def __init__(self, encoder: Zipformer, encoder_proj: nn.Linear):
        super().__init__()
        self.encoder = encoder
        self.encoder_proj = encoder_proj

    def forward(self, x: torch.Tensor, states: List[torch.Tensor]):
        n = x.size(0)
        t = x.size(1)
        x_lens = torch.tensor([t] * n, device=x.device)
        output, _, new_states = self.encoder.streaming_forward(
            x=x, x_lens=x_lens, states=states
        )
        return self.encoder_proj(output), new_states


class OnnxDecoder(nn.Module):
    def __init__(self, decoder: Decoder, decoder_proj: nn.Linear):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out = self.decoder(y, need_pad=False).squeeze(1)
        return self.decoder_proj(out)


class OnnxJoiner(nn.Module):
    def __init__(self, output_linear: nn.Linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        return self.output_linear(torch.tanh(encoder_out + decoder_out))


def add_metadata(path: Path, meta: Dict[str, str]) -> None:
    import onnx

    m = onnx.load(str(path))
    for k, v in meta.items():
        entry = m.metadata_props.add()
        entry.key = k
        entry.value = v
    onnx.save(m, str(path))


def simplify(src: Path, dst: Path) -> None:
    import onnx
    import onnxsim

    model = onnx.load(str(src))
    sim, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim failed for {src}")
    onnx.save(sim, str(dst))


def cut_reciprocal(path: Path) -> int:
    """Replace Reciprocal with 1/x Div (Pulsar2 friendly), in place."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    nodes = []
    changed = 0
    for node in graph.node:
        if node.op_type != "Reciprocal":
            nodes.append(node)
            continue
        changed += 1
        one_name = (node.name or "reciprocal") + "_one"
        one_tensor = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name)
        graph.initializer.append(one_tensor)
        div_node = helper.make_node(
            "Div", [one_name, node.input[0]], node.output,
            name=(node.name or "reciprocal") + "_div",
        )
        nodes.append(div_node)
    if changed:
        del graph.node[:]
        graph.node.extend(nodes)
        onnx.save(model, str(path))
    return changed


def export_encoder(enc: OnnxEncoder, out: Path, chunk_len: int, opset: int,
                   meta_cfg: dict) -> None:
    T = chunk_len + 7
    enc.encoder.__class__.forward = enc.encoder.__class__.streaming_forward
    x = torch.randn(1, T, 80, dtype=torch.float32) * 0.5
    init_state = enc.encoder.get_init_state()
    assert len(init_state) == 35, len(init_state)

    groups = [
        ("cached_len", init_state[0:5]),
        ("cached_avg", init_state[5:10]),
        ("cached_key", init_state[10:15]),
        ("cached_val", init_state[15:20]),
        ("cached_val2", init_state[20:25]),
        ("cached_conv1", init_state[25:30]),
        ("cached_conv2", init_state[30:35]),
    ]
    input_names = ["x"]
    output_names = ["encoder_out"]
    for name, tensors in groups:
        for i, s in enumerate(tensors):
            input_names.append(f"{name}_{i}")
            output_names.append(f"new_{name}_{i}")
    torch.onnx.export(
        enc, (x, init_state), str(out),
        verbose=False, opset_version=opset,
        input_names=input_names, output_names=output_names,
        do_constant_folding=True,
    )
    add_metadata(out, {
        "model_type": "zipformer",
        "version": "1",
        "model_author": "k2-fsa",
        "decode_chunk_len": str(chunk_len),
        "T": str(T),
        "num_encoder_layers": ",".join(map(str, meta_cfg["num_encoder_layers"])),
        "encoder_dims": ",".join(map(str, meta_cfg["encoder_dims"])),
        "attention_dims": ",".join(map(str, meta_cfg["attention_dims"])),
        "cnn_module_kernels": ",".join(map(str, meta_cfg["cnn_module_kernels"])),
        "left_context_len": ",".join(str(v) for v in meta_cfg["left_context_len"]),
    })


def export_decoder(dec: OnnxDecoder, out: Path, context_size: int,
                   vocab_size: int, opset: int) -> None:
    y = torch.zeros(1, context_size, dtype=torch.int64)
    torch.onnx.export(
        dec, y, str(out), verbose=False, opset_version=opset,
        input_names=["y"], output_names=["decoder_out"],
        do_constant_folding=True,
    )
    add_metadata(out, {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    })


def export_joiner(join: OnnxJoiner, out: Path, joiner_dim: int, opset: int) -> None:
    e = torch.randn(1, joiner_dim, dtype=torch.float32)
    d = torch.randn(1, joiner_dim, dtype=torch.float32)
    torch.onnx.export(
        join, (e, d), str(out), verbose=False, opset_version=opset,
        input_names=["encoder_out", "decoder_out"], output_names=["logit"],
        do_constant_folding=True,
    )
    add_metadata(out, {"joiner_dim": str(joiner_dim)})


def main() -> None:
    args = get_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"chunk{args.chunk_len}"

    blank_id, tokens_vocab, _ = parse_tokens(args.tokens)
    sd = load_state_dict(args.pt_model)

    vocab_size = int(sd["decoder.embedding.weight"].shape[0])
    decoder_dim = int(sd["decoder.embedding.weight"].shape[1])
    encoder_dim = int(sd["joiner.encoder_proj.weight"].shape[1])
    joiner_dim = int(sd["joiner.encoder_proj.weight"].shape[0])
    print(f"checkpoint: vocab={vocab_size} decoder_dim={decoder_dim} "
          f"encoder_dim={encoder_dim} joiner_dim={joiner_dim}")
    print(f"tokens.txt: blank_id={blank_id} vocab={tokens_vocab}")
    assert vocab_size == tokens_vocab, (vocab_size, tokens_vocab)

    encoder = Zipformer(
        num_features=80,
        output_downsampling_factor=2,
        zipformer_downsampling_factors=ENCODER_CFG["zipformer_downsampling_factors"],
        encoder_dims=ENCODER_CFG["encoder_dims"],
        attention_dim=ENCODER_CFG["attention_dims"],
        encoder_unmasked_dims=ENCODER_CFG["encoder_unmasked_dims"],
        nhead=ENCODER_CFG["nhead"],
        feedforward_dim=ENCODER_CFG["feedforward_dims"],
        cnn_module_kernels=ENCODER_CFG["cnn_module_kernels"],
        num_encoder_layers=ENCODER_CFG["num_encoder_layers"],
        num_left_chunks=ENCODER_CFG["num_left_chunks"],
        short_chunk_size=ENCODER_CFG["short_chunk_size"],
        decode_chunk_size=args.chunk_len // 2,
    )
    decoder = Decoder(vocab_size=vocab_size, decoder_dim=decoder_dim,
                      blank_id=blank_id, context_size=2)
    joiner = Joiner(encoder_dim=encoder_dim, decoder_dim=decoder_dim,
                    joiner_dim=joiner_dim, vocab_size=vocab_size)
    model = Model(encoder, decoder, joiner, encoder_dim, decoder_dim, vocab_size)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith(("simple_am_proj", "simple_lm_proj"))]
    bad_unexpected = [k for k in unexpected if not k.startswith(("simple_am_proj", "simple_lm_proj"))]
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if bad_missing or bad_unexpected:
        print("BAD missing:", bad_missing[:20])
        print("BAD unexpected:", bad_unexpected[:20])
        raise SystemExit("checkpoint does not match this streaming zipformer recipe")
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True)
    model.eval()

    meta_cfg = dict(ENCODER_CFG)
    meta_cfg["left_context_len"] = [
        encoder.left_context_len // ds for ds in ENCODER_CFG["zipformer_downsampling_factors"]
    ]
    print("left_context_len:", meta_cfg["left_context_len"])

    enc_wrap = OnnxEncoder(model.encoder, model.joiner.encoder_proj)
    dec_wrap = OnnxDecoder(model.decoder, model.joiner.decoder_proj)
    join_wrap = OnnxJoiner(model.joiner.output_linear)
    enc_wrap.eval(); dec_wrap.eval(); join_wrap.eval()

    files = {
        "encoder": out_dir / f"encoder-{prefix}.onnx",
        "decoder": out_dir / f"decoder-{prefix}.onnx",
        "joiner": out_dir / f"joiner-{prefix}.onnx",
    }
    export_encoder(enc_wrap, files["encoder"], args.chunk_len, args.opset, meta_cfg)
    export_decoder(dec_wrap, files["decoder"], 2, vocab_size, args.opset)
    export_joiner(join_wrap, files["joiner"], joiner_dim, args.opset)

    sim_files = {}
    for name, path in files.items():
        if args.no_simplify:
            sim = path
        else:
            sim = out_dir / f"{name}-{prefix}.sim.onnx"
            simplify(path, sim)
        if name == "encoder" and not args.no_cut_reciprocal:
            n = cut_reciprocal(sim)
            print(f"cut Reciprocal nodes: {n}")
        sim_files[name] = sim

    enc_cfg_out = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in ENCODER_CFG.items()
    }
    cfg = {
        "model": "zipformer_streaming",
        "checkpoint": str(Path(args.pt_model).resolve()),
        "tokens": str(Path(args.tokens).resolve()),
        "decode_chunk_len": args.chunk_len,
        "segment_frames": args.chunk_len + 7,
        "chunk_shift": args.chunk_len,
        **enc_cfg_out,
        "left_context_len": meta_cfg["left_context_len"],
        "encoder_dim": encoder_dim,
        "decoder_dim": decoder_dim,
        "joiner_dim": joiner_dim,
        "context_size": 2,
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "onnx": {k: str(v.resolve()) for k, v in sim_files.items()},
    }
    (out_dir / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
