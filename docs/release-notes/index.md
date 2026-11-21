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

## Unreleased

- Rescale eligible negative weights before weighted triplet sampling so
  finite double weights cannot erase all triplets when converted to float32.
  Exact seeded weighted draws may change; relative probabilities are retained.

## Published releases

## Unreleased

- Preserve higher-precision distance matrices in MultiSupConLoss when
  embeddings use a lower precision, retaining score differences, loss values,
  and matrix gradients.

## Published releases

## Unreleased

- Keep BatchEasyHardMiner and BatchHardMiner selections inside their candidate
  sets when finite scores reach a dtype limit, instead of selecting a masked
  self-comparison or wrong-label pair on a sentinel tie.

## Published releases

## Unreleased

Preserve NCA and ProxyNCA losses and gradients when finite distances or similarities carry large common row offsets.

## Published releases

## Unreleased

Keep RankedList weighted averages and gradients finite at large positive and negative temperatures, including low-precision inputs.

## Published releases

## Unreleased

Preserve the PNP variant objectives and gradients when well-separated examples have very small soft negative ranks.

## Published releases

## Unreleased

Handle empty retrieval queries consistently across self-search, custom label comparisons, NDCG, and per-class reductions.

## Published releases

## Unreleased

Keep SoftTriple center weights, class scores, and training gradients finite when temperature scaling would overflow before softmax.

## Published releases

## Unreleased

Avoid premature half-precision overflow in NormalizedSoftmax logits while retaining their public absolute scale and trainable class gradients.

## Published releases

## Unreleased

Preserve DynamicArcFace power-law margins for extreme finite powers and closely spaced integer class counts.

## Published releases

## Unreleased

Preserve valid high-dimensional mining anchors by widening low-precision distances before the density logarithm is evaluated.

## Published releases

## Unreleased

Preserve DCL and DCLW objectives when a common score offset would erase the negative partition difference or overflow before normalization.

## Published releases

## Unreleased

Compute low-precision intra-pair variance means and squared hinges in float32 so rounding cannot erase or amplify training gradients.

## Published releases

## Unreleased

Compute NDCG with shared gain scaling so finite large or subnormal relevance gains retain their ranking ratio.

## Published releases

## Unreleased

Form P2SGrad cosine residuals before low-precision rounding can erase otherwise representable embedding and parameter gradients.

## Published releases

## Unreleased

Widen CircleLoss adaptive-weight arithmetic so finite logits and gradients survive intermediate half-precision overflow.

## Published releases

## Unreleased

Normalize finite large embeddings without overflowing intermediate p-norm calculations or losing training gradients.

## Published releases

## Unreleased

- `BatchEasyHardMiner` now applies `allowed_pos_range` and `allowed_neg_range`
  when the corresponding strategy is `"all"`, preserving the inclusive bounds
  used by the other strategies.

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
