# Unreleased

## Numerical fixes

- `LargeMarginSoftmaxLoss` and `SphereFaceLoss` compute low-precision
  magnitude norms and scaling in float32 before returning logits in their
  existing result dtype. Representable scaled logits and training gradients
  are preserved in cases where intermediate norms or products previously
  overflowed or underflowed. Float64 precision, autocast result promotion,
  and the final logit dtype's range remain unchanged.
