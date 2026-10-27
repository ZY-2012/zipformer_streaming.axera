# Encoder quantization variants (AX650 board)

All variants use the same static ONNX (chunk 32 / T 39), decoder FP32 and
joiner U16.  Token error is measured against the fp32 ONNX hypotheses.

| variant | calib | smooth | extra FP32 | size | carry enc_out | token err 8 | token err 20 |
|---|---|---|---|---|---|---|---|
| v256 | 256 | yes | - | 72 MB | 0.9421 | 2.65% (7/264) | n/a |
| A | 1024 | yes | - | 72 MB | 0.9360 | 2.27% (6/264) | 0.735% (2/272) |
| B | 1024 | no | - | 72 MB | 0.9360 | 2.27% (6/264) | 0.735% (2/272) |
| D | 1024 | yes | attention key/value FC | 79 MB | **0.9460** | 2.27% (6/264) | 0.735% (2/272) |
| E | 1024 | no | attention key/value + val2 FC, Mul | 81 MB | 0.9332 | 2.27% (6/264) | 0.735% (2/272) |
| C | - | - | Conv FP32 | failed | - | - | - |

- C failed: AX650 float Conv only supports group == 1; the depthwise conv
  (group=384) cannot be FP32.
- A and B produce identical board outputs.
- D has the best state-carry encoder_out; token outputs are identical to A/B/E
  on these wavs.
- `new_cached_conv2_*` (~0.826) remains the lowest state output (U16 depthwise
  conv limitation); `new_cached_val2_*` (~0.786 on chunk0) is also low.
- 8-wav errors: 3.wav 4 tokens, 46.wav 1, 8k.wav 1; 20 in-domain wavs: 2/272.

Final choice: **D** (accuracy-first, best state carry, 79 MB).
