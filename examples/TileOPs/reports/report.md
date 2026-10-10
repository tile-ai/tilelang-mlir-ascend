# TileOPs Evaluation Report: all

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-09T02:46:19.876784+00:00 |
| evaluation_scope | all |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261009_022152_829230_all |
| git_commit | 84519be447f469806322c473e5f33fdf298ab519 |
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
| 100.0% | 7 | 112 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AdaLayerNormFwdOp | 100.0% (22/22) | 1.13e-02 | 5 | 0.41% – 69.38% |
| ArgmaxFwdOp | 100.0% (42/42) | 0.00e+00 | 5 | 5.44% – 41.39% |
| LerpTensorFwdOp | 100.0% (8/8) | 5.21e-08 | 7 | 67.38% – 80.61% |
| LogSumExpFwdOp | 100.0% (23/23) | 3.67e-08 | 6 | 3.23% – 36.61% |
| MishFwdOp | 100.0% (7/7) | 2.45e-05 | 6 | 54.01% – 64.87% |
| MultiHeadAttentionFwdOp | 100.0% (6/6) | 8.54e-04 | 8 | 14.34% – 23.44% |
| SSDChunkScanFwdOp | 100.0% (4/4) | 2.19e-04 | 11 | 0.13% – 20.30% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AdaLayerNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| dit-xl-2 | 8.9200 | 43.73 | {"m": 1024, "n": 1152} | float16 | msprof | main | 1.8000 |
| dit-xl-2 | 9.2600 | 42.13 | {"m": 1024, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 40.3000 | 69.38 | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 41.1000 | 68.03 | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-decode | 3.4000 | 0.41 | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.8000 |

### ArgmaxFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| lm-head-argmax | 8.3600 | 5.44 | [4, 102400] | float16 | msprof | argreduce_partial | 1.8000 |
| lm-head-argmax | 8.9400 | 5.72 | [4, 102400] | bfloat16 | msprof | argreduce_partial | 1.8000 |
| hidden-state-argmax | 22.5200 | 41.39 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-argmax | 44.7400 | 21.42 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| 3d-non-last-axis-argmax | 12.7600 | 22.10 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |

### LerpTensorFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 72.7200 | 76.90 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 69.3800 | 80.61 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 154.8000 | 77.13 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 392.2800 | 68.56 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 386.4200 | 69.82 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1729.0200 | 67.38 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1727.1801 | 67.44 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |

### LogSumExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| attn-weights-4k | 16.3800 | 28.45 | [32, 32, 4096] | float16 | msprof | main | 1.8000 |
| attn-weights-4k | 17.2600 | 27.09 | [32, 32, 4096] | bfloat16 | msprof | main | 1.8000 |
| attn-weights-32k | 101.8400 | 36.61 | [32, 32, 32768] | bfloat16 | msprof | main | 1.8000 |
| lm-head-logits | 14.1000 | 3.23 | [4, 102400] | float16 | msprof | main | 1.8000 |
| lm-head-logits | 13.8000 | 3.30 | [4, 102400] | bfloat16 | msprof | main | 1.8000 |
| 3d-multidim-reduce | 10.4200 | 22.36 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |

### MishFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| yolo-p3 | 77.6000 | 61.28 | [16, 256, 80, 80] | float16 | msprof | main | 1.8000 |
| yolo-p3 | 80.6200 | 64.87 | [16, 256, 80, 80] | bfloat16 | msprof | main | 1.8000 |
| yolo-p4 | 40.9400 | 58.08 | [16, 512, 40, 40] | float16 | msprof | main | 1.8000 |
| yolo-p4 | 41.7200 | 62.68 | [16, 512, 40, 40] | bfloat16 | msprof | main | 1.8000 |
| fc-wide | 28.1800 | 54.01 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| fc-wide | 29.0200 | 57.67 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |

### MultiHeadAttentionFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-3.1-8b-short | 287.9200 | 14.62 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-short | 293.7600 | 14.34 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 981.4800 | 23.44 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 1038.0400 | 22.19 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 291.4800 | 14.44 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 289.7600 | 14.54 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 997.5000 | 23.07 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 1035.7400 | 22.24 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |

### SSDChunkScanFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| b1-c2-L64-h4-p64-n32-fp16 | 31.8200 | 0.13 | {"batch": 1, "chunk_len": 64, "d_head": 64, "d_state": 32, "n_groups": 1, "n_heads": 4, "num_chunks": 2} | float16 | msprof | main | 1.8000 |
| b2-c4-L64-h8-p64-n64-fp16 | 35.1400 | 1.31 | {"batch": 2, "chunk_len": 64, "d_head": 64, "d_state": 64, "n_groups": 2, "n_heads": 8, "num_chunks": 4} | float16 | msprof | main | 1.8000 |
| b1-c2-L128-h4-p128-n32-bf16 | 34.3000 | 0.29 | {"batch": 1, "chunk_len": 128, "d_head": 128, "d_state": 32, "n_groups": 1, "n_heads": 4, "num_chunks": 2} | bfloat16 | msprof | main | 1.8000 |
| b2-c2-L64-h4-p64-n32-bf16 | 32.7600 | 0.30 | {"batch": 2, "chunk_len": 64, "d_head": 64, "d_state": 32, "n_groups": 2, "n_heads": 4, "num_chunks": 2} | bfloat16 | msprof | main | 1.8000 |
| latency-130m-4k | 112.0600 | 9.38 | {"batch": 1, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| serving-130m-4k | 740.2200 | 14.11 | {"batch": 8, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| longctx-130m-32k | 2906.8198 | 20.30 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 128} | float16 | msprof | main | 1.8000 |
| latency-2p7b-4k | 303.4200 | 11.55 | {"batch": 1, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| serving-2p7b-4k | 1233.1400 | 16.36 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| longctx-2p7b-32k | 4913.5601 | 20.08 | {"batch": 2, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 128} | float16 | msprof | main | 1.8000 |
| throughput-2p7b-2k | 582.1400 | 12.16 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 8} | float16 | msprof | main | 1.8000 |
