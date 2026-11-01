# Additional GPU timing measurements

3 fresh processes per configuration, each averaging 3 blocks of 10 forward/backward pairs after warmup. Times are milliseconds per pair; peak memory is total PyTorch allocation in MiB. Encoder, optimizer, and data loading are excluded. This campaign measures timing and memory; full-gradient agreement and capacity require separate evidence.

| GPU            | GPU memory (GiB) | Host CPU                            | PyTorch      | CUDA |
| -------------- | ---------------: | ----------------------------------- | ------------ | ---- |
| RTX 4090       |            23.52 | AMD EPYC 7352 24-Core Processor     | 2.14.0+cu130 | 13.0 |
| RTX 5090       |            31.36 | AMD Ryzen 9 9950X 16-Core Processor | 2.14.0+cu130 | 13.0 |
| L40S           |            44.39 | AMD EPYC 9354 32-Core Processor     | 2.14.0+cu130 | 13.0 |
| A100-SXM4-80GB |            79.25 | AMD EPYC 7742 64-Core Processor     | 2.14.0+cu130 | 13.0 |
| H100 80GB HBM3 |            79.18 | Intel(R) Xeon(R) Platinum 8470      | 2.14.0+cu130 | 13.0 |

[CSV measurements](measurements.csv) retain full precision and statuses.

standard: 128 dimensions, 4 samples per class for pair/triplet losses, 1,024 classes for proxy losses.

## All-triplet · standard · float32 · batch 8,192

| GPU            | Version | Status | Total | Forward | Backward | Process range | Peak MiB |
| -------------- | ------- | ------ | ----: | ------: | -------: | ------------- | -------: |
| RTX 4090       | 2.9.0   | ok     | 3,041 |   2,448 |      593 | 3,027–3,058   |   16,528 |
| RTX 4090       | 3.0.0   | ok     |   289 |    91.6 |      198 | 289–289       |   15,950 |
| RTX 5090       | 2.9.0   | ok     |   838 |     619 |      219 | 837–840       |   16,566 |
| RTX 5090       | 3.0.0   | ok     |   194 |    54.6 |      139 | 194–194       |   15,989 |
| L40S           | 2.9.0   | ok     | 1,530 |   1,161 |      368 | 1,477–1,557   |   16,528 |
| L40S           | 3.0.0   | ok     |   375 |     126 |      249 | 375–375       |   15,950 |
| A100-SXM4-80GB | 2.9.0   | ok     | 2,540 |   2,137 |      402 | 2,532–2,553   |   16,528 |
| A100-SXM4-80GB | 3.0.0   | ok     |   293 |    54.9 |      238 | 293–293       |   15,950 |
| H100 80GB HBM3 | 2.9.0   | ok     | 1,316 |   1,110 |      206 | 1,273–1,392   |   16,566 |
| H100 80GB HBM3 | 3.0.0   | ok     |   158 |    32.4 |      126 | 158–158       |   15,989 |

## Raw evidence

RTX 4090: [raw records](gpu-1-raw.jsonl.gz) · [configuration and hardware](gpu-1-provenance.json)

RTX 5090: [raw records](gpu-2-raw.jsonl.gz) · [configuration and hardware](gpu-2-provenance.json)

L40S: [raw records](gpu-3-raw.jsonl.gz) · [configuration and hardware](gpu-3-provenance.json)

A100-SXM4-80GB: [raw records](gpu-4-raw.jsonl.gz) · [configuration and hardware](gpu-4-provenance.json)

H100 80GB HBM3: [raw records](gpu-5-raw.jsonl.gz) · [configuration and hardware](gpu-5-provenance.json)
