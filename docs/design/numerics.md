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

Dynamic soft margin weights distance differences by a running cumulative
histogram. Histogram updates are detached statistics, not part of the gradient
graph. Normalize each batch histogram before applying momentum so a larger
triplet count does not silently override that parameter. Bin centers include
both endpoints of `[min_val, -min_val]`; out-of-range observations saturate their
weights at the nearest endpoint without clipping the loss's distance difference.
The histogram is a registered buffer and survives checkpointing and device moves.
Matched-pair mining masks a separate distance tensor and never mutates the
distance output saved by autograd. Unequal streams contribute only real matched
diagonal pairs; their extra rows remain eligible negatives. Empty valid sets
leave histogram state unchanged.

This contract follows the histogram and detached-weight algorithm from Zhang
et al., [Learning Local Descriptors With a CDF-Based Dynamic Soft Margin](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf),
and the [authors' implementation](https://github.com/lg-zhang/dynamic-soft-margin-pytorch).
Domenico Muscillo contributed the original PML implementation. Independent tests
use triangular-basis probabilities and explicit distance formulas to verify
values, gradients, and sequential updates. Float16 histogram and gradient comparisons allow
`5e-4` absolute and `5e-3` relative error for rounding during normalization and
accumulation; float32 and float64 retain PyTorch's default comparison tolerances.

The mathematical definitions, supported arguments, and paper references remain
in `docs/losses.md`, `docs/miners.md`, `docs/distances.md`, `docs/reducers.md`,
and `docs/regularizers.md`. In particular, P2SGrad computes mean squared error
between cosine scores and one-hot labels, which gives the paper's specified
gradient. Reference implementations in the tests compare values and gradients;
their existing upstream attributions remain alongside those implementations.

The development baseline uses current PyTorch tensor operations. Compatibility
with obsolete PyTorch releases is not a requirement.
