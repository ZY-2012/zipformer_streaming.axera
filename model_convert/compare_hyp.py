#!/usr/bin/env python3
"""Compare two streaming hypotheses JSONs (tokens/text) without ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    hyp = json.loads(Path(args.hyp).read_text(encoding="utf-8"))

    rows = []
    total_err = total_len = 0
    for name in sorted(set(ref) & set(hyp)):
        a = ref[name]["tokens"]
        b = hyp[name]["tokens"]
        ed = edit_distance(a, b)
        total_err += ed
        total_len += len(a)
        rows.append((name, len(a), ed, ref[name]["text"] == hyp[name]["text"],
                     ref[name]["text"], hyp[name]["text"]))
    rate = total_err / max(total_len, 1) * 100
    lines = ["# Streaming hypotheses comparison (fp32 ONNX vs board AXMODEL)", "",
             f"- token error rate: **{rate:.2f}%** ({total_err}/{total_len})",
             "", "| wav | tokens | edit | text_equal |", "|---|---|---|---|"]
    for name, n, ed, eq, rt, ht in rows:
        lines.append(f"| {name} | {n} | {ed} | {eq} |")
    lines += ["", "## texts", ""]
    for name, n, ed, eq, rt, ht in rows:
        lines.append(f"- **{name}** ref: {rt}")
        lines.append(f"  hyp: {ht}")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"token error rate: {rate:.2f}% ({total_err}/{total_len})")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
