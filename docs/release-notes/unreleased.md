# Unreleased

## Numerical fixes

- `LpRegularizer` and `ZeroMeanRegularizer` compute low-precision statistics
  and return nonempty losses in float32. Large per-row penalties no longer
  overflow before averaging or applying a regularization weight, and squared
  Lp penalties preserve representable weighted gradients. Float32 and float64
  inputs retain their dtype; empty inputs retain their differentiable zero.
