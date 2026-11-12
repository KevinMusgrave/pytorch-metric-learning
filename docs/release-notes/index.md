# Release notes

## Unreleased

- `VICRegLoss` computes low-precision input statistics and returns its scalar
  in float32, with autocast disabled over the covariance computation. This
  prevents half-precision overflow and improves value and gradient accuracy.
  Float64 views retain their precision, including mixed-dtype view pairs.

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
