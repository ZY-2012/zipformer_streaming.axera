# C++ 板端推理

AXEngine + kaldi-native-fbank 的流式 Zipformer ASR，和 Python 版逻辑对齐：

- 16 kHz 单声道 16-bit PCM WAV
- 80 维 kaldi fbank（25 ms / 10 ms，dither=0，snip_edges=false，povey，
  low_freq=20，high_freq=-400）
- 流式 chunk=32 帧，encoder 输入 segment=39 帧（7 帧重叠），shift=32
- encoder → joiner/decoder greedy（context_size=2，blank_id=0）
- 尾部补 0.3 s 静音

## 目录

```
cpp/
├── zipformer_asr_ax.cpp
├── src/                       # AXEngine 封装 + WAV 读取
├── third_party/               # kaldi-native-fbank + kissfft 源码包
├── build_ax650.sh             # AX650 交叉编译
├── build_ax630c.sh            # AX630C 交叉编译（未验证）
├── download_toolchains.sh     # gcc 交叉编译器 + BSP SDK
├── CMakeLists.txt
└── bin/zipformer_asr_ax650    # 预编译 AX650 可执行文件
```

## 编译

```bash
# 下载 gcc 交叉编译器 + AX650 BSP（默认放到 cpp/toolchains/）
bash cpp/download_toolchains.sh

# 编译（产物 cpp/bin/zipformer_asr_ax650）
bash cpp/build_ax650.sh
```

已有工具链时：

```bash
export TOOLCHAIN_ROOT=<path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu>
export BSP_MSP_DIR=<path/to/ax650n_bsp_sdk/msp/out>
bash cpp/build_ax650.sh
```

## 运行（AX650N 板端）

```bash
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}

# 单条
./cpp/bin/zipformer_asr_ax650 \
  --models-dir models --tokens models/tokens.txt \
  --audio samples/0.wav

# 目录
./cpp/bin/zipformer_asr_ax650 \
  --models-dir models --tokens models/tokens.txt \
  --audio-dir samples
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models-dir` | `models` | encoder/decoder/joiner.axmodel |
| `--tokens` | `models/tokens.txt` | token 表 |
| `--audio` | - | 单个 16 kHz mono 16-bit PCM WAV |
| `--audio-dir` | `samples` | 批量处理目录下所有 `*.wav` |

输出包含每条音频的文本、耗时和 RTF；多文件时输出平均 RTF。

## 精度 / 速度

- C++ 与 Python 在 16 kHz 样本上输出一致；RTF ≈ 0.10–0.12（AX650N）。
- 模型加载约 0.7 s；encoder 80 MB，decoder 14 MB，joiner 3.3 MB。
