# Streaming zipformer static ONNX validation

## Ground truth: original pfluo TorchScript traces

- worst cosine: **1.0000000000** (threshold 0.999)
- chunk=32, T=39, batch=1, opset=18

| tensor | shape | cosine |
|---|---|---|
| decoder_out | (1, 512) | 1.0000000000 |
| decoder_out | (1, 512) | 1.0000000000 |
| decoder_out | (1, 512) | 1.0000000000 |
| decoder_out | (1, 512) | 1.0000000000 |
| decoder_out | (1, 512) | 1.0000000000 |
| encoder_out | (1, 8, 512) | 1.0000000000 |
| new_cached_len_0 | (2, 1) | 1.0000000000 |
| new_cached_len_1 | (4, 1) | 1.0000000000 |
| new_cached_len_2 | (3, 1) | 1.0000000000 |
| new_cached_len_3 | (2, 1) | 1.0000000000 |
| new_cached_len_4 | (4, 1) | 1.0000000000 |
| new_cached_avg_0 | (2, 1, 384) | 1.0000000000 |
| new_cached_avg_1 | (4, 1, 384) | 1.0000000000 |
| new_cached_avg_2 | (3, 1, 384) | 1.0000000000 |
| new_cached_avg_3 | (2, 1, 384) | 1.0000000000 |
| new_cached_avg_4 | (4, 1, 384) | 1.0000000000 |
| new_cached_key_0 | (2, 64, 1, 192) | 1.0000000000 |
| new_cached_key_1 | (4, 32, 1, 192) | 1.0000000000 |
| new_cached_key_2 | (3, 16, 1, 192) | 1.0000000000 |
| new_cached_key_3 | (2, 8, 1, 192) | 1.0000000000 |
| new_cached_key_4 | (4, 32, 1, 192) | 1.0000000000 |
| new_cached_val_0 | (2, 64, 1, 96) | 1.0000000000 |
| new_cached_val_1 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val_2 | (3, 16, 1, 96) | 1.0000000000 |
| new_cached_val_3 | (2, 8, 1, 96) | 1.0000000000 |
| new_cached_val_4 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val2_0 | (2, 64, 1, 96) | 1.0000000000 |
| new_cached_val2_1 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val2_2 | (3, 16, 1, 96) | 1.0000000000 |
| new_cached_val2_3 | (2, 8, 1, 96) | 1.0000000000 |
| new_cached_val2_4 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_conv1_0 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_1 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_2 | (3, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_3 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_4 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_0 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_1 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_2 | (3, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_3 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_4 | (4, 1, 384, 30) | 1.0000000000 |
| encoder_out (chunk2) | (1, 8, 512) | 1.0000000000 |
| new_cached_len_0 | (2, 1) | 1.0000000000 |
| new_cached_len_1 | (4, 1) | 1.0000000000 |
| new_cached_len_2 | (3, 1) | 1.0000000000 |
| new_cached_len_3 | (2, 1) | 1.0000000000 |
| new_cached_len_4 | (4, 1) | 1.0000000000 |
| new_cached_avg_0 | (2, 1, 384) | 1.0000000000 |
| new_cached_avg_1 | (4, 1, 384) | 1.0000000000 |
| new_cached_avg_2 | (3, 1, 384) | 1.0000000000 |
| new_cached_avg_3 | (2, 1, 384) | 1.0000000000 |
| new_cached_avg_4 | (4, 1, 384) | 1.0000000000 |
| new_cached_key_0 | (2, 64, 1, 192) | 1.0000000000 |
| new_cached_key_1 | (4, 32, 1, 192) | 1.0000000000 |
| new_cached_key_2 | (3, 16, 1, 192) | 1.0000000000 |
| new_cached_key_3 | (2, 8, 1, 192) | 1.0000000000 |
| new_cached_key_4 | (4, 32, 1, 192) | 1.0000000000 |
| new_cached_val_0 | (2, 64, 1, 96) | 1.0000000000 |
| new_cached_val_1 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val_2 | (3, 16, 1, 96) | 1.0000000000 |
| new_cached_val_3 | (2, 8, 1, 96) | 1.0000000000 |
| new_cached_val_4 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val2_0 | (2, 64, 1, 96) | 1.0000000000 |
| new_cached_val2_1 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val2_2 | (3, 16, 1, 96) | 1.0000000000 |
| new_cached_val2_3 | (2, 8, 1, 96) | 1.0000000000 |
| new_cached_val2_4 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_conv1_0 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_1 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_2 | (3, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_3 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_4 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_0 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_1 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_2 | (3, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_3 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_4 | (4, 1, 384, 30) | 1.0000000000 |
| encoder_out (chunk3) | (1, 8, 512) | 1.0000000000 |
| new_cached_len_0 | (2, 1) | 1.0000000000 |
| new_cached_len_1 | (4, 1) | 1.0000000000 |
| new_cached_len_2 | (3, 1) | 1.0000000000 |
| new_cached_len_3 | (2, 1) | 1.0000000000 |
| new_cached_len_4 | (4, 1) | 1.0000000000 |
| new_cached_avg_0 | (2, 1, 384) | 1.0000000000 |
| new_cached_avg_1 | (4, 1, 384) | 1.0000000000 |
| new_cached_avg_2 | (3, 1, 384) | 1.0000000000 |
| new_cached_avg_3 | (2, 1, 384) | 1.0000000000 |
| new_cached_avg_4 | (4, 1, 384) | 1.0000000000 |
| new_cached_key_0 | (2, 64, 1, 192) | 1.0000000000 |
| new_cached_key_1 | (4, 32, 1, 192) | 1.0000000000 |
| new_cached_key_2 | (3, 16, 1, 192) | 1.0000000000 |
| new_cached_key_3 | (2, 8, 1, 192) | 1.0000000000 |
| new_cached_key_4 | (4, 32, 1, 192) | 1.0000000000 |
| new_cached_val_0 | (2, 64, 1, 96) | 1.0000000000 |
| new_cached_val_1 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val_2 | (3, 16, 1, 96) | 1.0000000000 |
| new_cached_val_3 | (2, 8, 1, 96) | 1.0000000000 |
| new_cached_val_4 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val2_0 | (2, 64, 1, 96) | 1.0000000000 |
| new_cached_val2_1 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_val2_2 | (3, 16, 1, 96) | 1.0000000000 |
| new_cached_val2_3 | (2, 8, 1, 96) | 1.0000000000 |
| new_cached_val2_4 | (4, 32, 1, 96) | 1.0000000000 |
| new_cached_conv1_0 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_1 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_2 | (3, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_3 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv1_4 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_0 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_1 | (4, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_2 | (3, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_3 | (2, 1, 384, 30) | 1.0000000000 |
| new_cached_conv2_4 | (4, 1, 384, 30) | 1.0000000000 |
| logit | (1, 6254) | 1.0000000000 |
| logit | (1, 6254) | 1.0000000000 |
| logit | (1, 6254) | 1.0000000000 |

## Note: sherpa-onnx 2023-02-20 is a different checkpoint

The public sherpa-onnx model is described as coming from pfluo, but its
weights do not match the current pfluo `pretrained.pt` (the pfluo jit
traces match our export at cosine 1.0). Decoder initializer cosines:

- decoder.embedding.weight: 0.2516
- decoder.conv.weight: -0.2248
- decoder_proj.weight: 0.1857

Therefore the sherpa ONNX is not used as a weight-level reference.
