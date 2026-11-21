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

- `LargeMarginSoftmaxLoss` and `SphereFaceLoss` compute low-precision
  magnitude norms and scaling in float32 before returning logits in their
  existing result dtype. Representable scaled logits and training gradients
  are preserved in cases where intermediate norms or products previously
  overflowed or underflowed. Float64 precision, autocast result promotion,
  and the final logit dtype's range remain unchanged.

- Keep ManifoldLoss values and gradients finite for large finite margins by
  evaluating both ranking terms in log space.
- Preserve ArcFace target-cosine derivatives in the monotonic continuation
  and at zero margin, including SubCenterArcFace and DynamicArcFace training
  inputs whose target cosines round to an endpoint.
- Keep the reduced spherical embedding penalty in float32 for float16 and
  bfloat16 inputs, so later loss weighting cannot inherit premature overflow
  or underflow. Float32 and float64 output dtypes remain unchanged.
- Rescale eligible negative weights before weighted triplet sampling so
  finite double weights cannot erase all triplets when converted to float32.
  Exact seeded weighted draws may change; relative probabilities are retained.
- Preserve higher-precision distance matrices in MultiSupConLoss when
  embeddings use a lower precision, retaining score differences, loss values,
  and matrix gradients.
- Keep BatchEasyHardMiner and BatchHardMiner selections inside their candidate
  sets when finite scores reach a dtype limit, instead of selecting a masked
  self-comparison or wrong-label pair on a sentinel tie.
- Preserve NCA and ProxyNCA losses and gradients when finite distances or similarities carry large common row offsets.
- Keep RankedList weighted averages and gradients finite at large positive and negative temperatures, including low-precision inputs.
- Preserve the PNP variant objectives and gradients when well-separated examples have very small soft negative ranks.
- Handle empty retrieval queries consistently across self-search, custom label comparisons, NDCG, and per-class reductions.
- Keep SoftTriple center weights, class scores, and training gradients finite when temperature scaling would overflow before softmax.
- Avoid premature half-precision overflow in NormalizedSoftmax logits while retaining their public absolute scale and trainable class gradients.
- Preserve DynamicArcFace power-law margins for extreme finite powers and closely spaced integer class counts.
- Preserve valid high-dimensional mining anchors by widening low-precision distances before the density logarithm is evaluated.
- Preserve DCL and DCLW objectives when a common score offset would erase the negative partition difference or overflow before normalization.
- Compute low-precision intra-pair variance means and squared hinges in float32 so rounding cannot erase or amplify training gradients.
- Compute NDCG with shared gain scaling so finite large or subnormal relevance gains retain their ranking ratio.
- Form P2SGrad cosine residuals before low-precision rounding can erase otherwise representable embedding and parameter gradients.
- Widen CircleLoss adaptive-weight arithmetic so finite logits and gradients survive intermediate half-precision overflow.
- Normalize finite large embeddings without overflowing intermediate p-norm calculations or losing training gradients.
- `BatchEasyHardMiner` now applies `allowed_pos_range` and `allowed_neg_range`
  when the corresponding strategy is `"all"`, preserving the inclusive bounds
  used by the other strategies.

## Numerical composition

- Keep NormalizedSoftmax and Circle per-example terms in their accumulation
  dtype through reduction. Half-precision inputs therefore return float32
  nonempty losses, preserving finite means of large terms.
- Preserve the public NormalizedSoftmax `get_logits` extension path in training.
- Center score calculations without overflowing a finite span before a smaller
  multiplier or larger temperature restores its range. SoftTriple handles both
  temperature signs.
- Use the shared large-vector normalization in P2SGrad while retaining float32
  cosine residuals and the original input epsilon.

## Development dependencies

- Update certifi to the aged 2026.10.18 security fix, removing compromised emSign
  roots. Retire the expired certifi applicability and uv release-age assessments;
  uv 0.13.4 now satisfies the normal admission window.
