# TileOPs Evaluation Report: all

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-09-18T06:44:30.382877+00:00 |
| evaluation_scope | all |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20260918_062713_465732_all |
| git_commit | 1990aa9fe49ecf2a94ae00477581f257af7f953e |
| correctness_target | tests/ops |
| benchmark_target | benchmarks/ops |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 16 |
| cpu | x86_64 |
| cann | 8.5.0 |
| driver | 26.0.rc1 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1 |
| tilelang | 0.1.2+ubuntu.22.4.npuir |
| python | 3.11.15 |
| os | Linux-5.15.0-191-generic-x86_64-with-glibc2.35 |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 5 | 66 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AdaLayerNormFwdOp | 100.0% (22/22) | 1.13e-02 | 8 | 0.50% – 70.54% |
| LerpTensorFwdOp | 100.0% (8/8) | 5.21e-08 | 10 | 45.28% – 78.29% |
| LogSumExpFwdOp | 100.0% (23/23) | 3.67e-08 | 6 | 3.19% – 36.58% |
| MishFwdOp | 100.0% (7/7) | 2.45e-05 | 9 | 26.51% – 64.84% |
| MultiHeadAttentionFwdOp | 100.0% (6/6) | 8.54e-04 | 10 | 2.66% – 23.29% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AdaLayerNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-dit | 3.1000 | 15.86 | {"m": 64, "n": 1152} | float32 | msprof | main | 1.8000 |
| smoke-dit | 3.3000 | 7.45 | {"m": 64, "n": 1152} | float16 | msprof | main | 1.8000 |
| smoke-dit | 3.6000 | 6.83 | {"m": 64, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| dit-xl-2 | 9.3000 | 41.95 | {"m": 1024, "n": 1152} | float16 | msprof | main | 1.8000 |
| dit-xl-2 | 8.7800 | 44.43 | {"m": 1024, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 39.6400 | 70.54 | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 40.8400 | 68.47 | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-decode | 2.8000 | 0.50 | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.8000 |

### LerpTensorFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 10.9600 | 63.78 | [1024, 1024] | float32 | msprof | main | 1.8000 |
| smoke-1m | 7.4600 | 46.85 | [1024, 1024] | float16 | msprof | main | 1.8000 |
| smoke-1m | 7.7200 | 45.28 | [1024, 1024] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 71.7400 | 77.95 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 72.9200 | 76.69 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 152.8800 | 78.29 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 393.8200 | 68.69 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 388.7600 | 69.22 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1726.0800 | 67.47 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1731.9399 | 67.26 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |

### LogSumExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| attn-weights-4k | 16.3600 | 28.49 | [32, 32, 4096] | float16 | msprof | main | 1.8000 |
| attn-weights-4k | 16.8000 | 27.83 | [32, 32, 4096] | bfloat16 | msprof | main | 1.8000 |
| attn-weights-32k | 101.9200 | 36.58 | [32, 32, 32768] | bfloat16 | msprof | main | 1.8000 |
| lm-head-logits | 14.1600 | 3.22 | [4, 102400] | float16 | msprof | main | 1.8000 |
| lm-head-logits | 14.3000 | 3.19 | [4, 102400] | bfloat16 | msprof | main | 1.8000 |
| 3d-multidim-reduce | 9.4600 | 24.63 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |

### MishFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 8.6600 | 28.55 | [1048576] | float32 | msprof | main | 1.8000 |
| smoke-1m | 7.1800 | 26.51 | [1048576] | float16 | msprof | main | 1.8000 |
| smoke-1m | 6.7800 | 30.87 | [1048576] | bfloat16 | msprof | main | 1.8000 |
| yolo-p3 | 75.4400 | 63.04 | [16, 256, 80, 80] | float16 | msprof | main | 1.8000 |
| yolo-p3 | 80.6600 | 64.84 | [16, 256, 80, 80] | bfloat16 | msprof | main | 1.8000 |
| yolo-p4 | 40.0600 | 59.36 | [16, 512, 40, 40] | float16 | msprof | main | 1.8000 |
| yolo-p4 | 41.5400 | 62.95 | [16, 512, 40, 40] | bfloat16 | msprof | main | 1.8000 |
| fc-wide | 28.3600 | 53.66 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| fc-wide | 27.9200 | 59.94 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |

### MultiHeadAttentionFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mha-fwd-smoke-s512-h8-d64 | 43.9400 | 2.66 | {"batch": 1, "causal": true, "dim": 64, "heads": 8, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| mha-fwd-smoke-s512-h8-d64 | 43.9600 | 2.66 | {"batch": 1, "causal": true, "dim": 64, "heads": 8, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-short | 286.4000 | 14.70 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-short | 291.3200 | 14.46 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 988.0600 | 23.29 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 1028.9000 | 22.38 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 288.6400 | 14.58 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 295.0600 | 14.28 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 1001.6600 | 22.97 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 1037.0000 | 22.21 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
