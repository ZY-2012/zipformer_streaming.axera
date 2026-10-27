# Board axmodel vs fp32 ONNX

- worst float cosine: **0.78568049**
- int state max diff: 0.0

| group | tensor | shape | cosine | int_max_diff |
|---|---|---|---|---|
| chunk0 | encoder_out | (1, 8, 512) | 0.99925911 | 0 |
| chunk0 | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk0 | new_cached_avg_0 | (2, 1, 384) | 0.96793680 | 0 |
| chunk0 | new_cached_avg_1 | (4, 1, 384) | 0.92163766 | 0 |
| chunk0 | new_cached_avg_2 | (3, 1, 384) | 0.88202605 | 0 |
| chunk0 | new_cached_avg_3 | (2, 1, 384) | 0.85635843 | 0 |
| chunk0 | new_cached_avg_4 | (4, 1, 384) | 0.92694349 | 0 |
| chunk0 | new_cached_key_0 | (2, 64, 1, 192) | 0.98302339 | 0 |
| chunk0 | new_cached_key_1 | (4, 32, 1, 192) | 0.97450213 | 0 |
| chunk0 | new_cached_key_2 | (3, 16, 1, 192) | 0.94367062 | 0 |
| chunk0 | new_cached_key_3 | (2, 8, 1, 192) | 0.93176651 | 0 |
| chunk0 | new_cached_key_4 | (4, 32, 1, 192) | 0.99155768 | 0 |
| chunk0 | new_cached_val_0 | (2, 64, 1, 96) | 0.94551347 | 0 |
| chunk0 | new_cached_val_1 | (4, 32, 1, 96) | 0.89282154 | 0 |
| chunk0 | new_cached_val_2 | (3, 16, 1, 96) | 0.85026109 | 0 |
| chunk0 | new_cached_val_3 | (2, 8, 1, 96) | 0.87298912 | 0 |
| chunk0 | new_cached_val_4 | (4, 32, 1, 96) | 0.91575800 | 0 |
| chunk0 | new_cached_val2_0 | (2, 64, 1, 96) | 0.95269860 | 0 |
| chunk0 | new_cached_val2_1 | (4, 32, 1, 96) | 0.91080170 | 0 |
| chunk0 | new_cached_val2_2 | (3, 16, 1, 96) | 0.78568049 | 0 |
| chunk0 | new_cached_val2_3 | (2, 8, 1, 96) | 0.85673554 | 0 |
| chunk0 | new_cached_val2_4 | (4, 32, 1, 96) | 0.92685107 | 0 |
| chunk0 | new_cached_conv1_0 | (2, 1, 384, 30) | 0.95907727 | 0 |
| chunk0 | new_cached_conv1_1 | (4, 1, 384, 30) | 0.86488772 | 0 |
| chunk0 | new_cached_conv1_2 | (3, 1, 384, 30) | 0.90785408 | 0 |
| chunk0 | new_cached_conv1_3 | (2, 1, 384, 30) | 0.84656601 | 0 |
| chunk0 | new_cached_conv1_4 | (4, 1, 384, 30) | 0.93261039 | 0 |
| chunk0 | new_cached_conv2_0 | (2, 1, 384, 30) | 0.96311743 | 0 |
| chunk0 | new_cached_conv2_1 | (4, 1, 384, 30) | 0.93226626 | 0 |
| chunk0 | new_cached_conv2_2 | (3, 1, 384, 30) | 0.88723840 | 0 |
| chunk0 | new_cached_conv2_3 | (2, 1, 384, 30) | 0.79080092 | 0 |
| chunk0 | new_cached_conv2_4 | (4, 1, 384, 30) | 0.97224522 | 0 |
| chunk1 (ONNX states) | encoder_out | (1, 8, 512) | 0.99992922 | 0 |
| chunk1 (ONNX states) | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk1 (ONNX states) | new_cached_avg_0 | (2, 1, 384) | 0.99365548 | 0 |
| chunk1 (ONNX states) | new_cached_avg_1 | (4, 1, 384) | 0.99570382 | 0 |
| chunk1 (ONNX states) | new_cached_avg_2 | (3, 1, 384) | 0.99812521 | 0 |
| chunk1 (ONNX states) | new_cached_avg_3 | (2, 1, 384) | 0.99628015 | 0 |
| chunk1 (ONNX states) | new_cached_avg_4 | (4, 1, 384) | 0.99418284 | 0 |
| chunk1 (ONNX states) | new_cached_key_0 | (2, 64, 1, 192) | 0.99856714 | 0 |
| chunk1 (ONNX states) | new_cached_key_1 | (4, 32, 1, 192) | 0.99904463 | 0 |
| chunk1 (ONNX states) | new_cached_key_2 | (3, 16, 1, 192) | 0.99917687 | 0 |
| chunk1 (ONNX states) | new_cached_key_3 | (2, 8, 1, 192) | 0.99819131 | 0 |
| chunk1 (ONNX states) | new_cached_key_4 | (4, 32, 1, 192) | 0.99984898 | 0 |
| chunk1 (ONNX states) | new_cached_val_0 | (2, 64, 1, 96) | 0.99470136 | 0 |
| chunk1 (ONNX states) | new_cached_val_1 | (4, 32, 1, 96) | 0.99629766 | 0 |
| chunk1 (ONNX states) | new_cached_val_2 | (3, 16, 1, 96) | 0.99732888 | 0 |
| chunk1 (ONNX states) | new_cached_val_3 | (2, 8, 1, 96) | 0.99665573 | 0 |
| chunk1 (ONNX states) | new_cached_val_4 | (4, 32, 1, 96) | 0.99809730 | 0 |
| chunk1 (ONNX states) | new_cached_val2_0 | (2, 64, 1, 96) | 0.99573119 | 0 |
| chunk1 (ONNX states) | new_cached_val2_1 | (4, 32, 1, 96) | 0.99702842 | 0 |
| chunk1 (ONNX states) | new_cached_val2_2 | (3, 16, 1, 96) | 0.99688777 | 0 |
| chunk1 (ONNX states) | new_cached_val2_3 | (2, 8, 1, 96) | 0.99373640 | 0 |
| chunk1 (ONNX states) | new_cached_val2_4 | (4, 32, 1, 96) | 0.99808056 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_0 | (2, 1, 384, 30) | 0.99674854 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_1 | (4, 1, 384, 30) | 0.99607352 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_2 | (3, 1, 384, 30) | 0.99845884 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_3 | (2, 1, 384, 30) | 0.99539602 | 0 |
| chunk1 (ONNX states) | new_cached_conv1_4 | (4, 1, 384, 30) | 0.99842136 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_0 | (2, 1, 384, 30) | 0.99712303 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_1 | (4, 1, 384, 30) | 0.99853384 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_2 | (3, 1, 384, 30) | 0.99857157 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_3 | (2, 1, 384, 30) | 0.99460997 | 0 |
| chunk1 (ONNX states) | new_cached_conv2_4 | (4, 1, 384, 30) | 0.99934997 | 0 |
| chunk1 (board state carry) | encoder_out | (1, 8, 512) | 0.94598994 | 0 |
| chunk1 (board state carry) | new_cached_len_0 | (2, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_1 | (4, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_2 | (3, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_3 | (2, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_len_4 | (4, 1) | 1.00000000 | 0 |
| chunk1 (board state carry) | new_cached_avg_0 | (2, 1, 384) | 0.96698250 | 0 |
| chunk1 (board state carry) | new_cached_avg_1 | (4, 1, 384) | 0.93704646 | 0 |
| chunk1 (board state carry) | new_cached_avg_2 | (3, 1, 384) | 0.92936607 | 0 |
| chunk1 (board state carry) | new_cached_avg_3 | (2, 1, 384) | 0.89302631 | 0 |
| chunk1 (board state carry) | new_cached_avg_4 | (4, 1, 384) | 0.95216660 | 0 |
| chunk1 (board state carry) | new_cached_key_0 | (2, 64, 1, 192) | 0.98778462 | 0 |
| chunk1 (board state carry) | new_cached_key_1 | (4, 32, 1, 192) | 0.98307164 | 0 |
| chunk1 (board state carry) | new_cached_key_2 | (3, 16, 1, 192) | 0.96416278 | 0 |
| chunk1 (board state carry) | new_cached_key_3 | (2, 8, 1, 192) | 0.94735221 | 0 |
| chunk1 (board state carry) | new_cached_key_4 | (4, 32, 1, 192) | 0.99448316 | 0 |
| chunk1 (board state carry) | new_cached_val_0 | (2, 64, 1, 96) | 0.96061419 | 0 |
| chunk1 (board state carry) | new_cached_val_1 | (4, 32, 1, 96) | 0.93613887 | 0 |
| chunk1 (board state carry) | new_cached_val_2 | (3, 16, 1, 96) | 0.90829393 | 0 |
| chunk1 (board state carry) | new_cached_val_3 | (2, 8, 1, 96) | 0.89712688 | 0 |
| chunk1 (board state carry) | new_cached_val_4 | (4, 32, 1, 96) | 0.94757668 | 0 |
| chunk1 (board state carry) | new_cached_val2_0 | (2, 64, 1, 96) | 0.96370433 | 0 |
| chunk1 (board state carry) | new_cached_val2_1 | (4, 32, 1, 96) | 0.94558885 | 0 |
| chunk1 (board state carry) | new_cached_val2_2 | (3, 16, 1, 96) | 0.86608077 | 0 |
| chunk1 (board state carry) | new_cached_val2_3 | (2, 8, 1, 96) | 0.86960910 | 0 |
| chunk1 (board state carry) | new_cached_val2_4 | (4, 32, 1, 96) | 0.95298253 | 0 |
| chunk1 (board state carry) | new_cached_conv1_0 | (2, 1, 384, 30) | 0.97501174 | 0 |
| chunk1 (board state carry) | new_cached_conv1_1 | (4, 1, 384, 30) | 0.91451487 | 0 |
| chunk1 (board state carry) | new_cached_conv1_2 | (3, 1, 384, 30) | 0.94203800 | 0 |
| chunk1 (board state carry) | new_cached_conv1_3 | (2, 1, 384, 30) | 0.87099037 | 0 |
| chunk1 (board state carry) | new_cached_conv1_4 | (4, 1, 384, 30) | 0.95113432 | 0 |
| chunk1 (board state carry) | new_cached_conv2_0 | (2, 1, 384, 30) | 0.97654438 | 0 |
| chunk1 (board state carry) | new_cached_conv2_1 | (4, 1, 384, 30) | 0.95508688 | 0 |
| chunk1 (board state carry) | new_cached_conv2_2 | (3, 1, 384, 30) | 0.92716946 | 0 |
| chunk1 (board state carry) | new_cached_conv2_3 | (2, 1, 384, 30) | 0.82594527 | 0 |
| chunk1 (board state carry) | new_cached_conv2_4 | (4, 1, 384, 30) | 0.97720782 | 0 |
| decoder/joiner | decoder_out | (1, 512) | 0.99997715 | 0 |
| decoder/joiner | logit | (1, 6254) | 0.99999925 | 0 |
