# Zipformer-Streaming.AXERA

Zipformer 流式中英双语 ASR 在 Axera AX650N 上的部署工程：
静态 ONNX 导出 → 真实流式状态校准 → Pulsar2 U16 量化 → Python/C++ 板端推理。

- [x] 静态 ONNX 导出（chunk=32，T=39，batch=1）+ pfluo jit 对分 cosine 1.0
- [x] 真实 streaming 状态校准集（1000+ chunk）
- [x] Pulsar2 7.0 量化：U16 激活 + S8 权重
- [x] Python 板端推理（`python/`，numpy + kaldi-native-fbank + axengine）
- [x] C++ 板端推理（`cpp/`，AXEngine + kaldi-native-fbank）
- [x] 板端 token 精度 / RTF 评测

量化模型与预编译二进制在 HuggingFace：
[HY-2012/zipformer_streaming.axera](https://huggingface.co/HY-2012/zipformer_streaming.axera)

上游模型：
[pfluo/k2fsa-zipformer-chinese-english-mixed](https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed)
（训练 recipe：icefall `pruned_transducer_stateless7_streaming`）

> 注意：`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` 是另一份
> 较旧的 pfluo checkpoint，与本工程权重不同；本工程严格使用 pfluo 原始权重。

## 结果（AX650N，chunk=32 / 320 ms）

与 FP32 ONNX 对比（无标注集，按 token 序列统计）：

| 测试集 | token error | 说明 |
|--------|-------------|------|
| 20 条域内 wav | **0.735%**（2/272） | 2 条各 1 个 token 不同 |
| 8 条公开测试 wav | **2.27%**（6/264） | 主要为 3.wav 尾部 4 个 token |

逐张量 / 板端 cosine：

| 项 | cosine |
|----|--------|
| encoder_out（新状态） | 0.9992 |
| encoder_out（状态跨 chunk 回传） | 0.946 |
| decoder_out | 0.99998 |
| joiner logit | 0.999999 |

速度（C++，16 kHz mono WAV，6 条样本）：

| 指标 | 数值 |
|------|------|
| RTF（含 fbank） | **0.103 – 0.121**（平均 0.114） |
| 模型加载 | ~0.7 s |
| 模型大小 | encoder 80 MB + decoder 14 MB + joiner 3.3 MB |

## 支持平台

| 平台 | Python | C++ | 状态 |
|------|--------|-----|------|
| AX650N | ✅ | ✅ | 板端实测 |
| AX630C | — | — | 未验证（Pulsar2 换 AX620E 目标后可复用源码） |

## 模型接口

流式 chunk = 32 帧 fbank，encoder 输入 T = 32 + 7 = 39 帧。

| 图 | 输入 | 输出 |
|----|------|------|
| encoder | `x [1,39,80] fp32` + 35 个状态 | `encoder_out [1,8,512] fp32` + 35 个新状态 |
| decoder | `y [1,2] int32` | `decoder_out [1,512] fp32` |
| joiner | `encoder_out [1,512]`, `decoder_out [1,512]` | `logit [1,6254] fp32` |

35 个状态：`cached_len/avg/key/val/val2/conv1/conv2`（每类 5 个 stage），
`cached_len` 为 int32，其余 fp32。blank_id=0，context_size=2，vocab=6254。

## 目录结构

```
Zipformer-Streaming.AXERA/
├── model_convert/       # ONNX 导出 → 对分 → 真实状态校准 → Pulsar2 量化
│   ├── export_onnx.py
│   ├── validate_onnx.py
│   ├── prepare_calib.py
│   ├── prepare_tasks.py
│   ├── compile_models.py
│   ├── configs/         # encoder / decoder / joiner Pulsar2 配置
│   ├── data/            # tokens.txt（上游 token 表）
│   └── README.md
├── python/              # Python 板端推理（axengine + kaldi-native-fbank）
├── cpp/                 # C++ 板端推理源码 + 交叉编译脚本 + 预编译 bin
│   └── bin/zipformer_asr_ax650
├── models/              # 仅 README，量化模型从 HuggingFace 下载
├── samples/             # 16 kHz 示例 wav
├── requirements.txt     # 板端 Python 依赖
└── LICENSE
```

## 环境

```bash
# 板端推理
pip install -r requirements.txt

# 转换（X86）
pip install -r model_convert/requirements.txt
```

Pulsar2 量化环境见 [model_convert/README.md](model_convert/README.md)。

## 快速开始（板端推理）

```bash
# 1) 从 HuggingFace 取量化模型与示例
hf download HY-2012/zipformer_streaming.axera --local-dir _hf_download
cp -r _hf_download/models .
cp -r _hf_download/samples .

# 2) Python（单条 / 目录）
python3 python/streaming_asr.py --model-dir models --tokens models/tokens.txt \
    --wav samples/0.wav --stats
python3 python/streaming_asr.py --model-dir models --tokens models/tokens.txt \
    --wav samples

# 3) C++（预编译 bin，板端直接运行）
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
./cpp/bin/zipformer_asr_ax650 --models-dir models --tokens models/tokens.txt \
    --audio samples/0.wav
./cpp/bin/zipformer_asr_ax650 --models-dir models --tokens models/tokens.txt \
    --audio-dir samples

# 或本地交叉编译（需要 gcc 交叉编译器 + AX650 BSP）
bash cpp/download_toolchains.sh
bash cpp/build_ax650.sh
```

C++ 只接受 16 kHz 单声道 16-bit PCM WAV；其他格式先重采样。

## 模型转换

完整步骤见 [model_convert/README.md](model_convert/README.md)：

```bash
# 1) 上游权重（约 300 MB）
hf download pfluo/k2fsa-zipformer-chinese-english-mixed --local-dir origin

# 2) 导出静态 ONNX + 与 pfluo TorchScript 对分
python model_convert/export_onnx.py \
  --pt-model origin/exp/pretrained.pt \
  --tokens origin/data/lang_char_bpe/tokens.txt \
  --output-dir build/onnx
python model_convert/validate_onnx.py

# 3) 真实 streaming 状态校准（准备 100+ 条 16 kHz wav）
python model_convert/prepare_calib.py \
  --encoder build/onnx/encoder-chunk32.sim.onnx \
  --decoder build/onnx/decoder-chunk32.sim.onnx \
  --joiner build/onnx/joiner-chunk32.sim.onnx \
  --wavs-dir <wav_dir> --output-dir build/calib --chunk-len 32 --t 39

# 4) 生成 Magnetar 任务并量化
python model_convert/prepare_tasks.py
MAGNETAR_ROOT=<path/to/Magnetar> python model_convert/compile_models.py \
  --models decoder,joiner,encoder
```

产物：`build/magnetar_tasks/<model>/compile/<model>.axmodel`。

### 量化结论

- 纯 INT8 在这些状态上精度不足；最终使用 **U16 激活 + S8 权重**。
- decoder 很小但输出敏感，保持 **FP32**。
- attention 的 key/value 投影设为 **FP32**（encoder 80 MB），state carry 最好。
- AX650 的 float Conv 只支持 `group=1`，depthwise conv（`group=384`）无法 FP32，
  只能保持 U16；这是目前 `cached_conv` 状态精度的硬件限制。
- 校准必须使用**真实流式状态**（逐 chunk 回填），不能用随机/全零状态。
- 误差主要集中在 3.wav 这类难句；20 条域内 wav 仅 2/272 token 不同。

## 限制

- 仅 greedy 解码，context_size=2。
- chunk=32（320 ms）流式；demo 为文件级推理，内部按 chunk 回填状态，不含 VAD/麦克风。
- C++ demo 仅支持 16 kHz mono 16-bit PCM WAV。
- AX630C 尚未验证。
- 本工程跟随 pfluo 原始 checkpoint；与 sherpa-onnx 2023-02-20 权重不同。

## 参考

- 上游模型：[pfluo/k2fsa-zipformer-chinese-english-mixed](https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed)
- icefall：[k2-fsa/icefall](https://github.com/k2-fsa/icefall)，recipe `pruned_transducer_stateless7_streaming`
- sherpa-onnx：[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- Pulsar2：[AXERA-TECH/Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2) · [文档](https://pulsar2-docs.readthedocs.io)
- 板端运行时：[AXERA-TECH/pyaxengine](https://github.com/AXERA-TECH/pyaxengine)
- AX650N BSP：[AXERA-TECH/ax650n_bsp_sdk](https://github.com/AXERA-TECH/ax650n_bsp_sdk)
- 工作流参考：[AXERA-TECH/Magnetar](https://github.com/AXERA-TECH/Magnetar)
- fbank：[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
- 部署模板：[ZY-2012/Opus-MT.AXERA](https://github.com/ZY-2012/Opus-MT.AXERA)

## License

本工程代码 Apache-2.0。模型权重遵循上游 pfluo / icefall 的许可。
