# Board axmodel vs fp32 ONNX

- worst float cosine: **0.78512462**
- int state max diff: 0.0

| group | tensor | shape | cosine | int_max_diff |
|---|---|---|---|---|
| chunk0 | encoder_out | (1, 8, 512) | 0.99924161 | 0 |
| chunk0 | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_avg_0 | (2, 1, 384) | 0.96809589 | 0 |
| chunk0 | new_cached_avg_1 | (4, 1, 384) | 0.92178475 | 0 |
| chunk0 | new_cached_avg_2 | (3, 1, 384) | 0.88200086 | 0 |
| chunk0 | new_cached_avg_3 | (2, 1, 384) | 0.85713461 | 0 |
| chunk0 | new_cached_avg_4 | (4, 1, 384) | 0.92524424 | 0 |
| chunk0 | new_cached_key_0 | (2, 64, 1, 192) | 0.98301646 | 0 |
| chunk0 | new_cached_key_1 | (4, 32, 1, 192) | 0.97448672 | 0 |
| chunk0 | new_cached_key_2 | (3, 16, 1, 192) | 0.94363395 | 0 |
| chunk0 | new_cached_key_3 | (2, 8, 1, 192) | 0.93161443 | 0 |
| chunk0 | new_cached_key_4 | (4, 32, 1, 192) | 0.99149453 | 0 |
| chunk0 | new_cached_val_0 | (2, 64, 1, 96) | 0.94558106 | 0 |
| chunk0 | new_cached_val_1 | (4, 32, 1, 96) | 0.89264932 | 0 |
| chunk0 | new_cached_val_2 | (3, 16, 1, 96) | 0.84983319 | 0 |
| chunk0 | new_cached_val_3 | (2, 8, 1, 96) | 0.87315765 | 0 |
| chunk0 | new_cached_val_4 | (4, 32, 1, 96) | 0.91476268 | 0 |
| chunk0 | new_cached_val2_0 | (2, 64, 1, 96) | 0.95272902 | 0 |
| chunk0 | new_cached_val2_1 | (4, 32, 1, 96) | 0.91075037 | 0 |
| chunk0 | new_cached_val2_2 | (3, 16, 1, 96) | 0.78512462 | 0 |
| chunk0 | new_cached_val2_3 | (2, 8, 1, 96) | 0.85655620 | 0 |
| chunk0 | new_cached_val2_4 | (4, 32, 1, 96) | 0.92593966 | 0 |
| chunk0 | new_cached_conv1_0 | (2, 1, 384, 30) | 0.95907790 | 0 |
| chunk0 | new_cached_conv1_1 | (4, 1, 384, 30) | 0.86488758 | 0 |
| chunk0 | new_cached_conv1_2 | (3, 1, 384, 30) | 0.90784477 | 0 |
| chunk0 | new_cached_conv1_3 | (2, 1, 384, 30) | 0.84695372 | 0 |
| chunk0 | new_cached_conv1_4 | (4, 1, 384, 30) | 0.93196500 | 0 |
| chunk0 | new_cached_conv2_0 | (2, 1, 384, 30) | 0.96316968 | 0 |
| chunk0 | new_cached_conv2_1 | (4, 1, 384, 30) | 0.93202735 | 0 |
| chunk0 | new_cached_conv2_2 | (3, 1, 384, 30) | 0.88740341 | 0 |
| chunk0 | new_cached_conv2_3 | (2, 1, 384, 30) | 0.79066195 | 0 |
| chunk0 | new_cached_conv2_4 | (4, 1, 384, 30) | 0.97196119 | 0 |
| chunk1 (ONNX states) | encoder_out | (1, 8, 512) | 0.99992999 | 0 |
| chunk1 (ONNX states) | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_avg_0 | (2, 1, 384) | 0.99381402 | 0 |
| chunk1 (ONNX states) | new_cached_avg_1 | (4, 1, 384) | 0.99580838 | 0 |
| chunk1 (ONNX states) | new_cached_avg_2 | (3, 1, 384) | 0.99790769 | 0 |
| chunk1 (ONNX states) | new_cached_avg_3 | (2, 1, 384) | 0.99616811 | 0 |
| chunk1 (ONNX states) | new_cached_avg_4 | (4, 1, 384) | 0.99380032 | 0 |
| chunk1 (ONNX states) | new_cached_key_0 | (2, 64, 1, 192) | 0.99850861 | 0 |
| chunk1 (ONNX states) | new_cached_key_1 | (4, 32, 1, 192) | 0.99899760 | 0 |
| chunk1 (ONNX states) | new_cached_key_2 | (3, 16, 1, 192) | 0.99911870 | 0 |
| chunk1 (ONNX states) | new_cached_key_3 | (2, 8, 1, 192) | 0.99806805 | 0 |
| chunk1 (ONNX states) | new_cached_key_4 | (4, 32, 1, 192) | 0.99983938 | 0 |
| chunk1 (ONNX states) | new_cached_val_0 | (2, 64, 1, 96) | 0.99457468 | 0 |
| chunk1 (ONNX states) | new_cached_val_1 | (4, 32, 1, 96) | 0.99610363 | 0 |
| chunk1 (ONNX states) | new_cached_val_2 | (3, 16, 1, 96) | 0.99717955 | 0 |
| chunk1 (ONNX states) | new_cached_val_3 | (2, 8, 1, 96) | 0.99642669 | 0 |
| chunk1 (ONNX states) | new_cached_val_4 | (4, 32, 1, 96) | 0.99797638 | 0 |
| chunk1 (ONNX states) | new_cached_val2_0 | (2, 64, 1, 96) | 0.99561671 | 0 |
| chunk1 (ONNX states) | new_cached_val2_1 | (4, 32, 1, 96) | 0.99690497 | 0 |
| chunk1 (ONNX states) | new_cached_val2_2 | (3, 16, 1, 96) | 0.99668414 | 0 |
| chunk1 (ONNX states) | new_cached_val2_3 | (2, 8, 1, 96) | 0.99357843 | 0 |
| chunk1 (ONNX states) | new_cached_val2_4 | (4, 32, 1, 96) | 0.99799437 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_0 | (2, 1, 384, 30) | 0.99664382 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_1 | (4, 1, 384, 30) | 0.99590943 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_2 | (3, 1, 384, 30) | 0.99833524 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_3 | (2, 1, 384, 30) | 0.99509722 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_4 | (4, 1, 384, 30) | 0.99831328 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_0 | (2, 1, 384, 30) | 0.99702190 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_1 | (4, 1, 384, 30) | 0.99847036 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_2 | (3, 1, 384, 30) | 0.99847814 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_3 | (2, 1, 384, 30) | 0.99433577 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_4 | (4, 1, 384, 30) | 0.99932494 | 0 |
| chunk1 (board state carry) | encoder_out | (1, 8, 512) | 0.93602838 | 0 |
| chunk1 (board state carry) | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_avg_0 | (2, 1, 384) | 0.96687218 | 0 |
| chunk1 (board state carry) | new_cached_avg_1 | (4, 1, 384) | 0.93692296 | 0 |
| chunk1 (board state carry) | new_cached_avg_2 | (3, 1, 384) | 0.92866378 | 0 |
| chunk1 (board state carry) | new_cached_avg_3 | (2, 1, 384) | 0.89349639 | 0 |
| chunk1 (board state carry) | new_cached_avg_4 | (4, 1, 384) | 0.95100672 | 0 |
| chunk1 (board state carry) | new_cached_key_0 | (2, 64, 1, 192) | 0.98775556 | 0 |
| chunk1 (board state carry) | new_cached_key_1 | (4, 32, 1, 192) | 0.98302917 | 0 |
| chunk1 (board state carry) | new_cached_key_2 | (3, 16, 1, 192) | 0.96406810 | 0 |
| chunk1 (board state carry) | new_cached_key_3 | (2, 8, 1, 192) | 0.94732011 | 0 |
| chunk1 (board state carry) | new_cached_key_4 | (4, 32, 1, 192) | 0.99442509 | 0 |
| chunk1 (board state carry) | new_cached_val_0 | (2, 64, 1, 96) | 0.96054488 | 0 |
| chunk1 (board state carry) | new_cached_val_1 | (4, 32, 1, 96) | 0.93597495 | 0 |
| chunk1 (board state carry) | new_cached_val_2 | (3, 16, 1, 96) | 0.90793441 | 0 |
| chunk1 (board state carry) | new_cached_val_3 | (2, 8, 1, 96) | 0.89730317 | 0 |
| chunk1 (board state carry) | new_cached_val_4 | (4, 32, 1, 96) | 0.94675217 | 0 |
| chunk1 (board state carry) | new_cached_val2_0 | (2, 64, 1, 96) | 0.96369210 | 0 |
| chunk1 (board state carry) | new_cached_val2_1 | (4, 32, 1, 96) | 0.94550811 | 0 |
| chunk1 (board state carry) | new_cached_val2_2 | (3, 16, 1, 96) | 0.86558705 | 0 |
| chunk1 (board state carry) | new_cached_val2_3 | (2, 8, 1, 96) | 0.86954410 | 0 |
| chunk1 (board state carry) | new_cached_val2_4 | (4, 32, 1, 96) | 0.95223306 | 0 |
| chunk1 (board state carry) | new_cached_conv1_0 | (2, 1, 384, 30) | 0.97495059 | 0 |
| chunk1 (board state carry) | new_cached_conv1_1 | (4, 1, 384, 30) | 0.91435465 | 0 |
| chunk1 (board state carry) | new_cached_conv1_2 | (3, 1, 384, 30) | 0.94192221 | 0 |
| chunk1 (board state carry) | new_cached_conv1_3 | (2, 1, 384, 30) | 0.87145328 | 0 |
| chunk1 (board state carry) | new_cached_conv1_4 | (4, 1, 384, 30) | 0.95037156 | 0 |
| chunk1 (board state carry) | new_cached_conv2_0 | (2, 1, 384, 30) | 0.97651584 | 0 |
| chunk1 (board state carry) | new_cached_conv2_1 | (4, 1, 384, 30) | 0.95491305 | 0 |
| chunk1 (board state carry) | new_cached_conv2_2 | (3, 1, 384, 30) | 0.92712362 | 0 |
| chunk1 (board state carry) | new_cached_conv2_3 | (2, 1, 384, 30) | 0.82604183 | 0 |
| chunk1 (board state carry) | new_cached_conv2_4 | (4, 1, 384, 30) | 0.97516227 | 0 |
| decoder/joiner | decoder_out | (1, 512) | 0.99997715 | 0 |
| decoder/joiner | logit | (1, 6254) | 0.99999925 | 0 |
