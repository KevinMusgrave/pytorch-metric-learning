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

Cross-batch memory stores detached reference embeddings. Each forward uses a
snapshot of the occupied embeddings and labels so subsequent queue writes cannot
invalidate tensors saved for backward. This supports accumulating several
forward losses before backpropagating. Only the occupied portion is copied while
the queue is warming up. An empty enqueue leaves the queue and its filled state
unchanged; an empty query yields a differentiable zero. Self-comparison removal
with no current queue indices removes nothing. A call cannot enqueue more rows
than the queue capacity, including when an enqueue mask is supplied.
The queue cursor and filled flag belong to PyTorch module extra state so a
`state_dict` checkpoint restores the next loss and gradients, not just the raw
buffers. The checkpoint contains tensors and primitive metadata and supports
`torch.load(..., weights_only=True)`. Older checkpoints without that metadata
do not describe a resumable queue and are not silently interpreted as one.

The mathematical definitions, supported arguments, and paper references remain
in `docs/losses.md`, `docs/miners.md`, `docs/distances.md`, `docs/reducers.md`,
and `docs/regularizers.md`. In particular, P2SGrad computes mean squared error
between cosine scores and one-hot labels, which gives the paper's specified
gradient. Reference implementations in the tests compare values and gradients;
their existing upstream attributions remain alongside those implementations.

The development baseline uses current PyTorch tensor operations. Compatibility
with obsolete PyTorch releases is not a requirement.
