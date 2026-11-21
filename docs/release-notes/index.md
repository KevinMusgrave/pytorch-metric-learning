# Release notes

- [Unreleased](unreleased.md)
## Unreleased

- Keep ManifoldLoss values and gradients finite for large finite margins by
  evaluating both ranking terms in log space.

## Published releases

## Unreleased

- Preserve ArcFace target-cosine derivatives in the monotonic continuation
  and at zero margin, including SubCenterArcFace and DynamicArcFace training
  inputs whose target cosines round to an endpoint.

## Published releases

## Unreleased

- Keep the reduced spherical embedding penalty in float32 for float16 and
  bfloat16 inputs, so later loss weighting cannot inherit premature overflow
  or underflow. Float32 and float64 output dtypes remain unchanged.

## Published releases

- [3.0.3](3.0.3.md)
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
