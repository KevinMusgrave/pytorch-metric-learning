# Numerical components

Losses, miners, distances, reducers, and regularizers share tensor conventions,
recordable statistics, and pair/triplet conversion. These mutually dependent
building blocks form one policy boundary. They must not depend on training,
dataset downloading, retrieval backends, or distributed process management.

Embeddings have shape `(batch, dimensions)`. Label equality defines positive
pairs; unequal labels define negative pairs. A pair tuple contains four index
tensors `(anchor_positive, positive, anchor_negative, negative)`, while a triplet
tuple contains `(anchor, positive, negative)`. Reference embeddings may have a
different batch size. Excluding self-comparisons depends on whether the query
and reference sets are the same set, not merely on equal shapes.

Distances declare whether larger values mean greater similarity. Losses and
miners use that declaration when comparing margins. Reducers own averaging and
filtering; losses retain the indices and reduction type needed by the reducer.
An empty valid set must yield a differentiable zero without introducing NaNs.

The mathematical definitions, supported arguments, and paper references remain
in `docs/losses.md`, `docs/miners.md`, `docs/distances.md`, `docs/reducers.md`,
and `docs/regularizers.md`. In particular, P2SGrad computes mean squared error
between cosine scores and one-hot labels, which gives the paper's specified
gradient. Reference implementations in the tests compare values and gradients;
their existing upstream attributions remain alongside those implementations.

The development baseline uses current PyTorch tensor operations. Compatibility
with obsolete PyTorch releases is not a requirement.
