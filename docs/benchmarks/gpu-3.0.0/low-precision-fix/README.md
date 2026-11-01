# Additional GPU timing measurements

3 fresh processes per configuration, each averaging 3 blocks of 100 forward/backward pairs after warmup. Times are milliseconds per pair; peak memory is total PyTorch allocation in MiB. Encoder, optimizer, and data loading are excluded. This campaign measures timing and memory; full-gradient agreement and capacity require separate evidence.

| GPU            | GPU memory (GiB) | Host CPU                            | PyTorch      | CUDA |
| -------------- | ---------------: | ----------------------------------- | ------------ | ---- |
| A100-SXM4-80GB |            79.25 | AMD EPYC 7742 64-Core Processor     | 2.14.0+cu130 | 13.0 |
| RTX 5090       |            31.36 | AMD Ryzen 9 7950X 16-Core Processor | 2.14.0+cu130 | 13.0 |

[CSV measurements](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v3.0.1/docs/benchmarks/gpu-3.0.0/low-precision-fix/measurements.csv) retain full precision and statuses.

standard: 128 dimensions, 4 samples per class for pair/triplet losses, 1,024 classes for proxy losses.

## Batch-hard + Triplet · standard · bfloat16 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  5.15 |    2.93 |     2.13 | 4.17–5.66     |     22.1 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.27 |    2.51 |     1.73 | 3.6–4.7       |      134 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  5.19 |    3.11 |        2 | 4.84–5.57     |     21.6 |
| RTX 5090       | 2.9.0             | ok     |  1.12 |   0.659 |    0.442 | 1.08–1.14     |     69.9 |
| RTX 5090       | 3.0.0             | ok     |   1.4 |   0.908 |     0.49 | 1.4–1.4       |      134 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.32 |   0.891 |    0.407 | 1.32–1.33     |     69.4 |

## Batch-hard + Triplet · standard · bfloat16 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.93 |     2.8 |     2.05 | 4.64–5.36     |      109 |
| A100-SXM4-80GB | 3.0.0             | ok     |    30 |      18 |     11.9 | 29.9–30       |    2,118 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  5.47 |    3.45 |     1.94 | 4.41–6.54     |      103 |
| RTX 5090       | 2.9.0             | ok     |   1.1 |   0.658 |    0.422 | 1.09–1.12     |      157 |
| RTX 5090       | 3.0.0             | ok     |  16.4 |    8.54 |     7.82 | 16.4–16.4     |    2,118 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.38 |    0.94 |     0.42 | 1.37–1.4      |      151 |

## Batch-hard + Triplet · standard · float32 · batch 128

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  5.31 |    3.07 |     2.15 | 5.07–5.48     |       17 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.17 |    2.39 |      1.7 | 4.02–4.25     |       17 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.73 |    2.91 |     1.74 | 4.27–5        |       17 |
| RTX 5090       | 2.9.0             | ok     |   1.1 |   0.651 |    0.426 | 1.08–1.14     |     64.8 |
| RTX 5090       | 3.0.0             | ok     |  1.01 |   0.652 |    0.338 | 1.01–1.01     |     64.8 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.02 |   0.655 |    0.347 | 1.01–1.05     |     64.8 |

## Batch-hard + Triplet · standard · float32 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  5.14 |    2.91 |     2.14 | 4.64–5.48     |     23.8 |
| A100-SXM4-80GB | 3.0.0             | ok     |   4.4 |    2.57 |     1.76 | 3.8–5.14      |     22.8 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.71 |    2.82 |      1.8 | 4.36–4.95     |     22.8 |
| RTX 5090       | 2.9.0             | ok     |  1.13 |   0.662 |     0.45 | 1.11–1.15     |     71.5 |
| RTX 5090       | 3.0.0             | ok     |  1.04 |   0.662 |    0.353 | 1.02–1.06     |     70.5 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.01 |   0.653 |     0.34 | 1.01–1.02     |     70.5 |

## Batch-hard + Triplet · standard · float32 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.93 |    2.75 |      2.1 | 4.62–5.46     |      133 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.54 |    2.66 |     1.79 | 4.08–5.07     |      117 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |   4.6 |    2.71 |     1.81 | 4.27–5.18     |      117 |
| RTX 5090       | 2.9.0             | ok     |  1.18 |   0.722 |    0.435 | 1.16–1.21     |      181 |
| RTX 5090       | 3.0.0             | ok     |  1.08 |   0.702 |    0.355 | 1.06–1.09     |      165 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.07 |     0.7 |    0.354 | 1.06–1.11     |      165 |

## Contrastive · standard · bfloat16 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.75 |    1.91 |     2.74 | 3.78–5.33     |     34.1 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.03 |    1.78 |     2.17 | 3.58–4.3      |      134 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |   4.9 |    2.45 |     2.37 | 4.56–5.08     |       35 |
| RTX 5090       | 2.9.0             | ok     |  1.03 |   0.437 |    0.574 | 1.01–1.07     |     81.8 |
| RTX 5090       | 3.0.0             | ok     |  1.04 |   0.441 |    0.599 | 1.04–1.04     |      134 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.01 |   0.537 |    0.454 | 0.993–1.03    |     82.7 |

## Contrastive · standard · bfloat16 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  7.63 |    1.89 |     5.74 | 7.57–7.71     |      299 |
| A100-SXM4-80GB | 3.0.0             | ok     |  23.4 |    9.08 |     14.3 | 23.4–23.4     |    2,118 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  6.32 |    1.91 |     4.41 | 5.97–6.61     |      308 |
| RTX 5090       | 2.9.0             | ok     |  2.56 |   0.429 |     2.13 | 2.56–2.57     |      346 |
| RTX 5090       | 3.0.0             | ok     |  13.3 |    4.29 |     9.03 | 13.3–13.3     |    2,118 |
| RTX 5090       | candidate-32bcfa8 | ok     |  2.23 |   0.535 |      1.7 | 2.23–2.24     |      356 |

## Contrastive · standard · float32 · batch 128

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.28 |    1.73 |     2.46 | 3.17–5.03     |     17.6 |
| A100-SXM4-80GB | 3.0.0             | ok     |  3.93 |    1.93 |     1.92 | 3.81–3.99     |     17.6 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  3.92 |     1.9 |     1.93 | 3.76–4.01     |     17.6 |
| RTX 5090       | 2.9.0             | ok     |  1.01 |   0.431 |    0.559 | 0.993–1.04    |     65.4 |
| RTX 5090       | 3.0.0             | ok     |  0.79 |   0.399 |    0.372 | 0.778–0.799   |     65.4 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.789 |   0.398 |    0.373 | 0.786–0.792   |     65.4 |

## Contrastive · standard · float32 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.63 |    1.84 |     2.69 | 3.48–5.29     |       36 |
| A100-SXM4-80GB | 3.0.0             | ok     |     4 |    1.92 |        2 | 3.67–4.25     |       36 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  3.28 |    1.54 |     1.66 | 2.75–4.32     |       36 |
| RTX 5090       | 2.9.0             | ok     |  1.02 |   0.433 |    0.565 | 1.01–1.03     |     83.7 |
| RTX 5090       | 3.0.0             | ok     | 0.822 |    0.41 |    0.393 | 0.805–0.854   |     83.7 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.808 |   0.408 |    0.382 | 0.8–0.814     |     83.7 |

## Contrastive · standard · float32 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  7.88 |    1.88 |     5.99 | 7.62–8.04     |      324 |
| A100-SXM4-80GB | 3.0.0             | ok     |  5.57 |    1.31 |     4.26 | 5.23–5.75     |      324 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  5.76 |    1.45 |     4.31 | 5.75–5.77     |      324 |
| RTX 5090       | 2.9.0             | ok     |  2.66 |   0.419 |     2.24 | 2.65–2.67     |      372 |
| RTX 5090       | 3.0.0             | ok     |  2.09 |   0.396 |     1.69 | 2.07–2.09     |      372 |
| RTX 5090       | candidate-32bcfa8 | ok     |  2.08 |   0.398 |     1.69 | 2.08–2.09     |      372 |

## NT-Xent · standard · bfloat16 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  16.5 |    10.6 |     5.92 | 16.5–16.6     |    3,451 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.69 |    2.31 |     2.28 | 4.31–4.9      |     33.9 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.68 |    2.27 |      2.3 | 4.42–4.9      |     33.9 |
| RTX 5090       | 2.9.0             | ok     |  13.3 |     7.8 |     5.55 | 13.3–13.4     |    3,498 |
| RTX 5090       | 3.0.0             | ok     | 0.932 |    0.47 |    0.444 | 0.923–0.944   |     81.6 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.942 |   0.469 |    0.453 | 0.917–0.973   |     81.6 |

## NT-Xent · standard · bfloat16 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | oom    |     — |       — |        — | —             |        — |
| A100-SXM4-80GB | 3.0.0             | ok     |  8.53 |    4.99 |     3.53 | 8.5–8.56      |      292 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  8.61 |    5.06 |     3.55 | 8.52–8.77     |      292 |
| RTX 5090       | 2.9.0             | oom    |     — |       — |        — | —             |        — |
| RTX 5090       | 3.0.0             | ok     |  3.16 |    1.63 |     1.53 | 3.15–3.16     |      339 |
| RTX 5090       | candidate-32bcfa8 | ok     |  3.16 |    1.63 |     1.53 | 3.15–3.16     |      339 |

## NT-Xent · standard · float32 · batch 128

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  3.58 |    1.46 |     2.04 | 2.63–4.13     |      119 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.27 |    2.06 |     2.11 | 3.91–4.52     |     17.5 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.01 |    1.91 |     2.01 | 3.31–4.56     |     17.5 |
| RTX 5090       | 2.9.0             | ok     | 0.781 |   0.337 |    0.426 | 0.771–0.793   |      166 |
| RTX 5090       | 3.0.0             | ok     |  0.87 |   0.438 |    0.414 | 0.848–0.91    |     65.3 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.894 |   0.444 |    0.429 | 0.864–0.913   |     65.3 |

## NT-Xent · standard · float32 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  24.5 |    14.6 |     9.91 | 24.5–24.5     |    6,500 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.66 |    2.26 |     2.29 | 4.59–4.71     |       35 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  3.88 |    1.86 |     1.93 | 3.64–4.09     |       35 |
| RTX 5090       | 2.9.0             | ok     |  24.6 |    13.9 |     10.7 | 24.6–24.6     |    6,548 |
| RTX 5090       | 3.0.0             | ok     | 0.875 |   0.442 |    0.413 | 0.869–0.886   |     82.7 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.866 |   0.438 |    0.408 | 0.862–0.872   |     82.7 |

## NT-Xent · standard · float32 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | oom    |     — |       — |        — | —             |        — |
| A100-SXM4-80GB | 3.0.0             | ok     |  8.47 |    4.74 |     3.72 | 8.42–8.51     |      308 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  8.54 |    4.81 |     3.72 | 8.5–8.59      |      308 |
| RTX 5090       | 2.9.0             | oom    |     — |       — |        — | —             |        — |
| RTX 5090       | 3.0.0             | ok     |  3.15 |    1.58 |     1.57 | 3.15–3.16     |      356 |
| RTX 5090       | candidate-32bcfa8 | ok     |  3.15 |    1.58 |     1.57 | 3.15–3.15     |      356 |

## ProxyAnchor · standard · bfloat16 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.18 |    2.21 |     1.88 | 3.91–4.66     |     33.3 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.93 |    2.46 |     2.38 | 4.1–5.75      |     25.8 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.16 |       2 |     2.08 | 3.44–4.71     |     25.8 |
| RTX 5090       | 2.9.0             | ok     | 0.933 |   0.528 |    0.381 | 0.929–0.937   |     80.3 |
| RTX 5090       | 3.0.0             | ok     |  1.06 |   0.554 |    0.481 | 1.03–1.1      |     73.5 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.08 |   0.559 |    0.492 | 1.03–1.1      |     73.5 |

## ProxyAnchor · standard · bfloat16 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.34 |    2.29 |     1.94 | 3.18–5.14     |     96.3 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.76 |    2.39 |     2.27 | 4.61–4.94     |     67.3 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.76 |    2.34 |     2.32 | 4.09–5.5      |     67.3 |
| RTX 5090       | 2.9.0             | ok     | 0.957 |   0.552 |    0.381 | 0.954–0.961   |      144 |
| RTX 5090       | 3.0.0             | ok     |  1.06 |    0.57 |     0.47 | 1.05–1.08     |      115 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.05 |   0.567 |    0.463 | 1.05–1.06     |      115 |

## ProxyAnchor · standard · float32 · batch 128

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.31 |    2.24 |     1.96 | 3.82–4.99     |     22.8 |
| A100-SXM4-80GB | 3.0.0             | ok     |  3.99 |    1.97 |     1.94 | 3.44–4.51     |     20.9 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.64 |    2.38 |     2.16 | 3.45–5.34     |     20.9 |
| RTX 5090       | 2.9.0             | ok     | 0.931 |   0.524 |    0.383 | 0.925–0.936   |     70.5 |
| RTX 5090       | 3.0.0             | ok     | 0.968 |   0.519 |    0.426 | 0.953–0.993   |     68.7 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.952 |   0.518 |    0.413 | 0.942–0.969   |     68.7 |

## ProxyAnchor · standard · float32 · batch 512

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.52 |     2.4 |     2.01 | 3.83–4.98     |     39.3 |
| A100-SXM4-80GB | 3.0.0             | ok     |  4.98 |    2.54 |     2.34 | 4.59–5.2      |     31.8 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.92 |    2.55 |     2.26 | 4.8–5.1       |     31.8 |
| RTX 5090       | 2.9.0             | ok     | 0.937 |   0.524 |    0.389 | 0.919–0.962   |     87.1 |
| RTX 5090       | 3.0.0             | ok     |  0.97 |   0.521 |    0.427 | 0.947–0.996   |     79.5 |
| RTX 5090       | candidate-32bcfa8 | ok     | 0.938 |    0.51 |    0.407 | 0.93–0.946    |     79.5 |

## ProxyAnchor · standard · float32 · batch 2,048

| GPU            | Version           | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ----------------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| A100-SXM4-80GB | 2.9.0             | ok     |  4.59 |    2.45 |     2.03 | 4.35–4.99     |      122 |
| A100-SXM4-80GB | 3.0.0             | ok     |  5.32 |    2.79 |     2.42 | 5.14–5.44     |     91.3 |
| A100-SXM4-80GB | candidate-32bcfa8 | ok     |  4.93 |    2.54 |     2.28 | 4.69–5.22     |     91.3 |
| RTX 5090       | 2.9.0             | ok     |  0.95 |   0.552 |    0.374 | 0.949–0.954   |      170 |
| RTX 5090       | 3.0.0             | ok     | 0.974 |    0.54 |    0.413 | 0.965–0.99    |      139 |
| RTX 5090       | candidate-32bcfa8 | ok     |  1.01 |    0.55 |    0.433 | 0.98–1.02     |      139 |

## Raw evidence

A100-SXM4-80GB: [raw records](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v3.0.1/docs/benchmarks/gpu-3.0.0/low-precision-fix/gpu-1-raw.jsonl.gz) · [configuration and hardware](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v3.0.1/docs/benchmarks/gpu-3.0.0/low-precision-fix/gpu-1-provenance.json)

RTX 5090: [raw records](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v3.0.1/docs/benchmarks/gpu-3.0.0/low-precision-fix/gpu-2-raw.jsonl.gz) · [configuration and hardware](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v3.0.1/docs/benchmarks/gpu-3.0.0/low-precision-fix/gpu-2-provenance.json)
