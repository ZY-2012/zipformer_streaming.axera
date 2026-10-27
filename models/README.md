# Models

本目录不存放量化模型（axmodel 约 97 MB，放在 HuggingFace）。

```bash
hf download HY-2012/zipformer_streaming.axera --local-dir _hf_download
cp -r _hf_download/models/* .
```

下载后包含：

```
encoder.axmodel   # streaming encoder, U16/S8 + attention key/value FP32
decoder.axmodel   # FP32
joiner.axmodel    # U16
tokens.txt        # 6254 token + <blk>
model_meta.json   # 三张图的输入/输出接口
config.json       # chunk=32, T=39, context=2, blank=0
```
