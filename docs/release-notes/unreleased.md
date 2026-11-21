# Unreleased

## Numerical fixes

- `SNRDistance` computes low-precision differences, variances, and their ratio
  in float32. Small nonzero variances no longer underflow before division,
  preventing finite float16 distances from becoming NaN and
  `SignalToNoiseRatioContrastiveLoss` from silently discarding their loss
  terms while propagating NaN gradients. The matrix and pairwise paths retain
  their existing result dtypes, and float64 inputs keep their precision.

- `LpRegularizer` and `ZeroMeanRegularizer` compute low-precision statistics
  and return nonempty losses in float32. Large per-row penalties no longer
  overflow before averaging or applying a regularization weight, and squared
  Lp penalties preserve representable weighted gradients. Float32 and float64
  inputs retain their dtype; empty inputs retain their differentiable zero.
