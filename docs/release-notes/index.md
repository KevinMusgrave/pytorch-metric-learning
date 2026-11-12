# Release notes

## Unreleased

- `UniformHistogramMiner` keeps distinct occupied bins for tiny distance
  ranges and uses float32 bin calculations for float16 and bfloat16 inputs.
  Positive and negative sampling now follows the actual observed range instead
  of merging bins through range clamping or low-precision rounding.

## Published versions

- [3.0.2](3.0.2.md)
- [3.0.1](3.0.1.md)
- [3.0.0](3.0.0.md)
- [3.0.0rc5 (preview)](3.0.0rc5.md)
- [3.0.0rc4 (preview)](3.0.0rc4.md)
- [3.0.0rc3 (preview)](3.0.0rc3.md)
- [3.0.0rc2 (preview)](3.0.0rc2.md)
- [3.0.0rc1 (preview)](3.0.0rc1.md)

Version 2 releases are listed on
[GitHub releases](https://github.com/KevinMusgrave/pytorch-metric-learning/releases),
and the [version 2 documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/v2/)
remains available.
