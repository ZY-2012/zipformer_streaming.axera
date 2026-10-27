# Python 板端推理

纯 Python 运行时：`numpy` + `axengine` + `kaldi-native-fbank`。

- 流式 encoder / decoder / joiner 三个 AXMODEL
- chunk=32，encoder 输入 T=39，35 个状态逐 chunk 回填
- 支持 WAV 输入（16 kHz mono 16-bit PCM）或预计算 fbank `.npy [T,80]`
- greedy 解码，context_size=2，blank=0

## 安装

```bash
pip install -r python/requirements.txt
```

`axengine` 安装见 [AXERA-TECH/pyaxengine](https://github.com/AXERA-TECH/pyaxengine)。

## 使用

```bash
# 单条 WAV
python3 python/streaming_asr.py \
  --model-dir models --tokens models/tokens.txt \
  --wav samples/0.wav --stats

# 目录（所有 *.wav）
python3 python/streaming_asr.py \
  --model-dir models --tokens models/tokens.txt \
  --wav samples --stats

# 预计算 fbank
python3 python/streaming_asr.py \
  --model-dir models --tokens models/tokens.txt \
  --features build/board_validate/features/0.npy

# 保存 JSON（tokens + text）
python3 python/streaming_asr.py \
  --model-dir models --tokens models/tokens.txt \
  --wav samples/0.wav --out result.json
```

`--stats` 打印模型加载时间、单条推理时间和 RTF。

## 作为库

```python
import sys
sys.path.insert(0, "python")
from streaming_asr import StreamingZipformer, read_wav, compute_fbank

asr = StreamingZipformer("models", "models/tokens.txt")
feats = compute_fbank(read_wav("samples/0.wav"))
print(asr.decode_features(feats)["text"])
```

`StreamingZipformer.reset()` 可复用同一个 session 处理多段音频。
