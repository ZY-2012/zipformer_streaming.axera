# 模型转换

pfluo streaming zipformer → 静态 ONNX → 真实状态校准 → Pulsar2 U16 量化。

## 环境

```bash
pip install -r model_convert/requirements.txt
```

Pulsar2 7.0 Docker 镜像，以及
[Magnetar](https://github.com/AXERA-TECH/Magnetar)（提供 `pulsar2 build` 封装）：

```bash
git clone https://github.com/AXERA-TECH/Magnetar.git
export MAGNETAR_ROOT=$PWD/Magnetar
```

## 1. 上游权重

```bash
hf download pfluo/k2fsa-zipformer-chinese-english-mixed --local-dir origin
```

需要的文件：

```
origin/exp/pretrained.pt
origin/exp/encoder_jit_trace.pt
origin/exp/decoder_jit_trace.pt
origin/exp/joiner_jit_trace.pt
origin/data/lang_char_bpe/tokens.txt
```

## 2. 导出静态 ONNX

```bash
python model_convert/export_onnx.py \
  --pt-model origin/exp/pretrained.pt \
  --tokens origin/data/lang_char_bpe/tokens.txt \
  --output-dir build/onnx --chunk-len 32
```

- chunk=32，encoder 输入 `T=39`（7 帧左上下文），35 个状态全部静态；
- 输出 `build/onnx/encoder-chunk32.sim.onnx`、`decoder-...`、`joiner-...`；
- `build/onnx/model_config.json` 记录 chunk / left context / vocab 等参数。

与 pfluo TorchScript 对分（cosine ≥ 0.999）：

```bash
python model_convert/validate_onnx.py
```

> `sherpa-onnx-...-2023-02-20` 是另一份旧 checkpoint，不作为权重级参考。

## 3. 真实状态校准

准备 100 条以上 16 kHz wav（建议 1–10 s，覆盖中英文）；逐 chunk 跑真实
encoder 并回填状态，保存每个输入张量的 `.zip`（Pulsar2 Numpy 校准格式）：

```bash
python model_convert/prepare_calib.py \
  --encoder build/onnx/encoder-chunk32.sim.onnx \
  --decoder build/onnx/decoder-chunk32.sim.onnx \
  --joiner build/onnx/joiner-chunk32.sim.onnx \
  --wavs-dir <wav_dir> \
  --output-dir build/calib \
  --chunk-len 32 --t 39
```

产物：`build/calib/x.zip`、`cached_*.zip`、`y.zip`、`encoder_out.zip`、
`decoder_out.zip` 等。

## 4. 生成 Magnetar 任务并量化

```bash
python model_convert/prepare_tasks.py
MAGNETAR_ROOT=<path/to/Magnetar> \
python model_convert/compile_models.py --models decoder,joiner,encoder
```

每个子模型一个 task（`build/magnetar_tasks/<model>/`）：

- `export/model.onnx` + `model_meta.json`
- `export/calib_data/*.zip`
- `compile/custom_config.json`（由 `prepare_tasks.py` 根据 ONNX 输入自动生成）
- `compile/<model>.axmodel`

Pulsar2 配置由 `prepare_tasks.py` 根据 ONNX 输入自动写入
`build/magnetar_tasks/<model>/compile/custom_config.json`（`/workspace/...`
是 Pulsar2 容器内路径）：

| 模型 | 配置 |
|------|------|
| encoder | U16 激活 + S8 权重；attention key/value 投影 FP32 |
| decoder | FP32（输出敏感） |
| joiner | U16 + S8 |

校准 `calibration_size=1024` 个真实 chunk。`enable_smooth_quant` 可开可关，
在本模型上端到端输出一致；默认关闭以缩短编译时间。

### 精度结论

- 纯 INT8 在 streaming 状态上精度不足；U16 激活 + S8 权重可满足要求。
- `decoder` 保持 FP32（14 MB），避免 joiner logits 偏移。
- attention 的 key/value 投影 FP32 可将 state carry encoder_out 从
  0.936 提升到 0.946；val2 投影 FP32（+`Mul` FP32）没有更好，未采用。
- AX650 的 float Conv 只支持 `group=1`；depthwise conv（`group=384`）不能
  FP32，只能保持 U16，这是 `cached_conv` 状态精度的主要限制。
- token error vs FP32 ONNX：20 条域内 wav 0.735%，8 条公开难例 2.27%。

## 5. 板端验证（可选）

```bash
# 主机：生成实际 chunk 与 ONNX 参考输出
python model_convert/make_board_cases.py \
  --onnx-dir build/onnx --out-dir build/board_validate --wavs samples/*.wav

# 板端：跑 axmodel
python3 python/board_cases.py \
  --model-dir <axmodel_dir> \
  --cases build/board_validate/cases.npz \
  --out build/board_validate/board_cases_out.npz

# 主机：对比
python model_convert/compare_board.py \
  --cases build/board_validate/cases.npz \
  --board build/board_validate/board_cases_out.npz \
  --report build/reports/board_cases.md
```

`compare_hyp.py` 可对比两份 `streaming_asr.py` 输出 JSON 的 token error。
