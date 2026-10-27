# Streaming Zipformer (zh-en) on AXERA AX650

End-to-end streaming (chunk-by-chunk) zipformer bilingual Chinese-English ASR
for AX650, quantized with Magnetar / Pulsar2 7.0.

- Source checkpoint: [pfluo/k2fsa-zipformer-chinese-english-mixed](https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed)
- Recipe: icefall `pruned_transducer_stateless7_streaming`
- Streaming: `decode_chunk_len=32`, encoder input `T=39` (7-frame left
  context), 35 carried states (`cached_len/avg/key/val/val2/conv1/conv2`)
- Static ONNX: batch=1, every input/output dim fixed (Pulsar2 requirement)
- Release: [HY-2012/zipformer_streaming.axera](https://huggingface.co/HY-2012/zipformer_streaming.axera)

## Checkpoint identity (important)

The current pfluo `pretrained.pt` and its TorchScript traces are internally
consistent; our static ONNX matches those traces at **cosine 1.0**.
The public `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` ONNX
does **not** contain the same weights (decoder embedding cosine ≈ 0.25); it is
an older/different pfluo checkpoint. This project follows the pfluo original
weights; the sherpa ONNX is only used as a structural reference.
Details: `reports/export_validate.md`.

## Pipeline

```
pretrained.pt
  └─ export/export_static.py        static ONNX (chunk32/T39, batch1)
       └─ export/validate_export.py  vs pfluo jit traces (cosine 1.0)
            └─ calib/gen_calib.py    real streaming-state calibration
                 └─ tools/prepare_magnetar_tasks.py
                      └─ tools/compile_magnetar.py  (Pulsar2 7.0)
                           └─ board_validate/*      (AX650 board)
```

Three submodels are compiled as separate Magnetar tasks because Pulsar2's
generic Magnetar pipeline is single-input; this project uses a custom
multi-input config and custom multi-model board validation.

## Quantization policy (accuracy-first)

| model | policy | size |
|---|---|---|
| encoder | U16 activations + S8 weights; attention key/value projections FP32 | 80 MB |
| decoder | FP32 | 14 MB |
| joiner | U16 | 3.3 MB |

AX650 does not support FP32 depthwise Conv (group=384), so the conv caches stay
U16; a fully FP32 encoder is not possible.

## Results (AX650, C++ runtime)

- encoder_out vs fp32 ONNX: 0.9992 (fresh states), 0.946 (state carry)
- decoder / joiner vs fp32 ONNX: 0.99998 / 0.999999
- token error vs fp32 ONNX:
  - 20 in-domain wavs: **0.735%** (2/272)
  - 8 public test wavs (harder): **2.27%** (6/264)
- RTF ≈ 0.10–0.12 (C++, 16 kHz WAV, 6 wavs)

## Reproduce

Host (Python 3.12 + torch 2.8 + onnx/onnxruntime/onnxsim):

```bash
python export/export_static.py --pt-model <pretrained.pt> \
  --tokens <tokens.txt> --output-dir export/onnx --chunk-len 32
python export/validate_export.py
python calib/gen_calib.py --encoder export/onnx/encoder-chunk32.sim.onnx \
  --decoder export/onnx/decoder-chunk32.sim.onnx \
  --joiner export/onnx/joiner-chunk32.sim.onnx \
  --wavs-dir <wavs> --output-dir calib/datasets --chunk-len 32 --t 39
python tools/prepare_magnetar_tasks.py
python tools/compile_magnetar.py --models decoder,joiner,encoder
```

Board validation (host + board share `/data/shared/huyuan`):

```bash
python tools/make_board_cases.py
# on board:
python3 runtime/python/board_cases.py --model-dir runtime/models_ax650 \
  --cases board_validate/cases.npz --out board_validate/board_cases_out.npz
python tools/compare_board.py --cases board_validate/cases.npz \
  --board board_validate/board_cases_out.npz --report reports/board_cases.md
```

## Layout

```
export/       static ONNX exporter + validation (k2/lhotse free)
calib/        real-state calibration generator
configs/      Pulsar2 configs (encoder U16+attention FP32, decoder FP32, joiner U16)
magnetar_tasks/  generated per-model Magnetar tasks (gitignored)
runtime/      Python board runtime + AX650 C++ runtime (kaldi-native-fbank)
tools/        task prep, compile runner, board case generation/compare
reports/      export validation, board cosine, hypotheses
release/      HF release bundle (gitignored; uploaded separately)
```
