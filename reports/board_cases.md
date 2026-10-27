# Board axmodel vs fp32 ONNX

- worst float cosine: **0.78672944**
- int state max diff: 0.0

| group | tensor | shape | cosine | int_max_diff |
|---|---|---|---|---|
| chunk0 | encoder_out | (1, 8, 512) | 0.99924075 | 0 |
| chunk0 | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_avg_0 | (2, 1, 384) | 0.97010992 | 0 |
| chunk0 | new_cached_avg_1 | (4, 1, 384) | 0.92666304 | 0 |
| chunk0 | new_cached_avg_2 | (3, 1, 384) | 0.87732555 | 0 |
| chunk0 | new_cached_avg_3 | (2, 1, 384) | 0.86247346 | 0 |
| chunk0 | new_cached_avg_4 | (4, 1, 384) | 0.93487567 | 0 |
| chunk0 | new_cached_key_0 | (2, 64, 1, 192) | 0.98303648 | 0 |
| chunk0 | new_cached_key_1 | (4, 32, 1, 192) | 0.97469103 | 0 |
| chunk0 | new_cached_key_2 | (3, 16, 1, 192) | 0.94380001 | 0 |
| chunk0 | new_cached_key_3 | (2, 8, 1, 192) | 0.93250162 | 0 |
| chunk0 | new_cached_key_4 | (4, 32, 1, 192) | 0.99158072 | 0 |
| chunk0 | new_cached_val_0 | (2, 64, 1, 96) | 0.94596326 | 0 |
| chunk0 | new_cached_val_1 | (4, 32, 1, 96) | 0.89326314 | 0 |
| chunk0 | new_cached_val_2 | (3, 16, 1, 96) | 0.85047249 | 0 |
| chunk0 | new_cached_val_3 | (2, 8, 1, 96) | 0.87536879 | 0 |
| chunk0 | new_cached_val_4 | (4, 32, 1, 96) | 0.91520126 | 0 |
| chunk0 | new_cached_val2_0 | (2, 64, 1, 96) | 0.95348276 | 0 |
| chunk0 | new_cached_val2_1 | (4, 32, 1, 96) | 0.91103235 | 0 |
| chunk0 | new_cached_val2_2 | (3, 16, 1, 96) | 0.78672944 | 0 |
| chunk0 | new_cached_val2_3 | (2, 8, 1, 96) | 0.85675510 | 0 |
| chunk0 | new_cached_val2_4 | (4, 32, 1, 96) | 0.92611228 | 0 |
| chunk0 | new_cached_conv1_0 | (2, 1, 384, 30) | 0.95939479 | 0 |
| chunk0 | new_cached_conv1_1 | (4, 1, 384, 30) | 0.86467937 | 0 |
| chunk0 | new_cached_conv1_2 | (3, 1, 384, 30) | 0.90892594 | 0 |
| chunk0 | new_cached_conv1_3 | (2, 1, 384, 30) | 0.84899293 | 0 |
| chunk0 | new_cached_conv1_4 | (4, 1, 384, 30) | 0.93201700 | 0 |
| chunk0 | new_cached_conv2_0 | (2, 1, 384, 30) | 0.96340692 | 0 |
| chunk0 | new_cached_conv2_1 | (4, 1, 384, 30) | 0.93294291 | 0 |
| chunk0 | new_cached_conv2_2 | (3, 1, 384, 30) | 0.88871206 | 0 |
| chunk0 | new_cached_conv2_3 | (2, 1, 384, 30) | 0.79090185 | 0 |
| chunk0 | new_cached_conv2_4 | (4, 1, 384, 30) | 0.97194487 | 0 |
| chunk1 (ONNX states) | encoder_out | (1, 8, 512) | 0.99993644 | 0 |
| chunk1 (ONNX states) | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_avg_0 | (2, 1, 384) | 0.99362132 | 0 |
| chunk1 (ONNX states) | new_cached_avg_1 | (4, 1, 384) | 0.99256850 | 0 |
| chunk1 (ONNX states) | new_cached_avg_2 | (3, 1, 384) | 0.98702475 | 0 |
| chunk1 (ONNX states) | new_cached_avg_3 | (2, 1, 384) | 0.99748075 | 0 |
| chunk1 (ONNX states) | new_cached_avg_4 | (4, 1, 384) | 0.99518799 | 0 |
| chunk1 (ONNX states) | new_cached_key_0 | (2, 64, 1, 192) | 0.99856298 | 0 |
| chunk1 (ONNX states) | new_cached_key_1 | (4, 32, 1, 192) | 0.99895581 | 0 |
| chunk1 (ONNX states) | new_cached_key_2 | (3, 16, 1, 192) | 0.99899923 | 0 |
| chunk1 (ONNX states) | new_cached_key_3 | (2, 8, 1, 192) | 0.99807383 | 0 |
| chunk1 (ONNX states) | new_cached_key_4 | (4, 32, 1, 192) | 0.99985466 | 0 |
| chunk1 (ONNX states) | new_cached_val_0 | (2, 64, 1, 96) | 0.99473244 | 0 |
| chunk1 (ONNX states) | new_cached_val_1 | (4, 32, 1, 96) | 0.99588910 | 0 |
| chunk1 (ONNX states) | new_cached_val_2 | (3, 16, 1, 96) | 0.99675393 | 0 |
| chunk1 (ONNX states) | new_cached_val_3 | (2, 8, 1, 96) | 0.99656826 | 0 |
| chunk1 (ONNX states) | new_cached_val_4 | (4, 32, 1, 96) | 0.99817794 | 0 |
| chunk1 (ONNX states) | new_cached_val2_0 | (2, 64, 1, 96) | 0.99572361 | 0 |
| chunk1 (ONNX states) | new_cached_val2_1 | (4, 32, 1, 96) | 0.99654368 | 0 |
| chunk1 (ONNX states) | new_cached_val2_2 | (3, 16, 1, 96) | 0.99638525 | 0 |
| chunk1 (ONNX states) | new_cached_val2_3 | (2, 8, 1, 96) | 0.99352400 | 0 |
| chunk1 (ONNX states) | new_cached_val2_4 | (4, 32, 1, 96) | 0.99816449 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_0 | (2, 1, 384, 30) | 0.99673051 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_1 | (4, 1, 384, 30) | 0.99543139 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_2 | (3, 1, 384, 30) | 0.99794555 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_3 | (2, 1, 384, 30) | 0.99528999 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_4 | (4, 1, 384, 30) | 0.99847542 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_0 | (2, 1, 384, 30) | 0.99704814 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_1 | (4, 1, 384, 30) | 0.99766652 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_2 | (3, 1, 384, 30) | 0.99825664 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_3 | (2, 1, 384, 30) | 0.99445189 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_4 | (4, 1, 384, 30) | 0.99940491 | 0 |
| chunk1 (board state carry) | encoder_out | (1, 8, 512) | 0.94212796 | 0 |
| chunk1 (board state carry) | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_avg_0 | (2, 1, 384) | 0.96934894 | 0 |
| chunk1 (board state carry) | new_cached_avg_1 | (4, 1, 384) | 0.94177713 | 0 |
| chunk1 (board state carry) | new_cached_avg_2 | (3, 1, 384) | 0.92264674 | 0 |
| chunk1 (board state carry) | new_cached_avg_3 | (2, 1, 384) | 0.90307070 | 0 |
| chunk1 (board state carry) | new_cached_avg_4 | (4, 1, 384) | 0.96072158 | 0 |
| chunk1 (board state carry) | new_cached_key_0 | (2, 64, 1, 192) | 0.98776410 | 0 |
| chunk1 (board state carry) | new_cached_key_1 | (4, 32, 1, 192) | 0.98322438 | 0 |
| chunk1 (board state carry) | new_cached_key_2 | (3, 16, 1, 192) | 0.96431981 | 0 |
| chunk1 (board state carry) | new_cached_key_3 | (2, 8, 1, 192) | 0.94817205 | 0 |
| chunk1 (board state carry) | new_cached_key_4 | (4, 32, 1, 192) | 0.99448915 | 0 |
| chunk1 (board state carry) | new_cached_val_0 | (2, 64, 1, 96) | 0.96086072 | 0 |
| chunk1 (board state carry) | new_cached_val_1 | (4, 32, 1, 96) | 0.93679609 | 0 |
| chunk1 (board state carry) | new_cached_val_2 | (3, 16, 1, 96) | 0.90886572 | 0 |
| chunk1 (board state carry) | new_cached_val_3 | (2, 8, 1, 96) | 0.89957835 | 0 |
| chunk1 (board state carry) | new_cached_val_4 | (4, 32, 1, 96) | 0.94706891 | 0 |
| chunk1 (board state carry) | new_cached_val2_0 | (2, 64, 1, 96) | 0.96452977 | 0 |
| chunk1 (board state carry) | new_cached_val2_1 | (4, 32, 1, 96) | 0.94576463 | 0 |
| chunk1 (board state carry) | new_cached_val2_2 | (3, 16, 1, 96) | 0.86702454 | 0 |
| chunk1 (board state carry) | new_cached_val2_3 | (2, 8, 1, 96) | 0.87049872 | 0 |
| chunk1 (board state carry) | new_cached_val2_4 | (4, 32, 1, 96) | 0.95230472 | 0 |
| chunk1 (board state carry) | new_cached_conv1_0 | (2, 1, 384, 30) | 0.97519463 | 0 |
| chunk1 (board state carry) | new_cached_conv1_1 | (4, 1, 384, 30) | 0.91443893 | 0 |
| chunk1 (board state carry) | new_cached_conv1_2 | (3, 1, 384, 30) | 0.94277183 | 0 |
| chunk1 (board state carry) | new_cached_conv1_3 | (2, 1, 384, 30) | 0.87399936 | 0 |
| chunk1 (board state carry) | new_cached_conv1_4 | (4, 1, 384, 30) | 0.95042896 | 0 |
| chunk1 (board state carry) | new_cached_conv2_0 | (2, 1, 384, 30) | 0.97685733 | 0 |
| chunk1 (board state carry) | new_cached_conv2_1 | (4, 1, 384, 30) | 0.95551676 | 0 |
| chunk1 (board state carry) | new_cached_conv2_2 | (3, 1, 384, 30) | 0.92816689 | 0 |
| chunk1 (board state carry) | new_cached_conv2_3 | (2, 1, 384, 30) | 0.82764136 | 0 |
| chunk1 (board state carry) | new_cached_conv2_4 | (4, 1, 384, 30) | 0.97650781 | 0 |
| decoder/joiner | decoder_out | (1, 512) | 0.99997715 | 0 |
| decoder/joiner | logit | (1, 6254) | 0.99999925 | 0 |
