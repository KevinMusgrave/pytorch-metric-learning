# Unreleased

## Numerical fixes

- `SNRDistance` computes low-precision differences, variances, and their ratio
  in float32. Small nonzero variances no longer underflow before division,
  preventing finite float16 distances from becoming NaN and
  `SignalToNoiseRatioContrastiveLoss` from silently discarding their loss
  terms while propagating NaN gradients. The matrix and pairwise paths retain
  their existing result dtypes, and float64 inputs keep their precision.
