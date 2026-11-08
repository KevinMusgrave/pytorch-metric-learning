# Release notes

## Unreleased

- `ClassWeightedReducer` now multiplies and averages low-precision losses in
  float32 before returning the scalar in the input dtype. This preserves
  representable weighted means and avoids prematurely rounding stored weights.
- `CenterInvariantRegularizer` now computes float16 and bfloat16 inputs in
  float32 and retains that dtype through reduction, preventing premature
  overflow and reducing rounding error in squared norm deviations.
- `RegularFaceRegularizer` now returns a differentiable zero for empty and
  singleton inputs instead of failing to select a neighbor or penalizing
  self-similarity.
- Samplers accept categorical NumPy labels and views with negative strides or
  non-native byte order. Invalid label dimensions and hierarchy columns now
  produce a descriptive error at construction.

## Releases

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
