#!/usr/bin/env python3
"""Run the per-model board cases (encoder chunk0/chunk1, decoder, joiner)
with axengine and dump all outputs for host-side comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from axengine import InferenceSession


def cast_states(states):
    return {
        k: (v.astype(np.int32) if k.startswith("cached_len") else v.astype(np.float32))
        for k, v in states.items()
    }


def run_encoder(sess, x, states):
    feed = {"x": np.ascontiguousarray(x, dtype=np.float32)}
    feed.update(cast_states(states))
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feed)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.cases, allow_pickle=False)
    md = Path(args.model_dir)
    enc = InferenceSession(str(md / "encoder.axmodel"))
    dec = InferenceSession(str(md / "decoder.axmodel"))
    join = InferenceSession(str(md / "joiner.axmodel"))

    state_names = [i.name for i in enc.get_inputs() if i.name != "x"]
    s0 = {k: d["s0_" + k] for k in state_names}
    s1 = {k: d["s1_" + k] for k in state_names}

    b0 = run_encoder(enc, d["x0"], s0)
    b1 = run_encoder(enc, d["x1"], s1)
    carry = {k: b0["new_" + k] for k in state_names}
    b1c = run_encoder(enc, d["x1"], carry)

    dec_in = dec.get_inputs()[0].name
    dec_out_name = dec.get_outputs()[0].name
    bd = dec.run([dec_out_name], {dec_in: d["y"].astype(np.int32)})[0]
    ji = [i.name for i in join.get_inputs()]
    jo = join.get_outputs()[0].name
    bj = join.run([jo], {ji[0]: d["enc_frame0"].astype(np.float32),
                         ji[1]: d["dec_out"].astype(np.float32)})[0]

    out = {}
    for k, v in b0.items():
        out["board0_" + k] = v
    for k, v in b1.items():
        out["board1_" + k] = v
    for k, v in b1c.items():
        out["board1carry_" + k] = v
    out["board_dec_out"] = bd
    out["board_logit"] = bj
    np.savez(args.out, **out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
