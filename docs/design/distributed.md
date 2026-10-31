# Distributed wrappers

Distributed wrappers gather reference embeddings and labels across ranks while
preserving the gradients and local-anchor semantics required by the wrapped
loss or miner. They depend on numerical components and process-group state,
not on the training orchestration classes.

Without a process group, or with an initialized one-rank group, wrappers delegate
to the wrapped component without gathering or scaling, and without warnings:
single-process debugging and evaluation are ordinary uses. Memory losses receive
`indices_tuple` and `enqueue_mask` through their own forward signature; ordinary
losses receive the query/reference arguments. Memory losses reject explicit
reference arguments, and ordinary losses reject enqueue masks. Validation precedes
memory updates. The absence of a process group must not bypass this dispatch.

The full-batch mode gathers tensors in rank order using PyTorch's autograd-aware
functional collective. Every rank computes the same global loss. The collective
sums gradients for each originating rank, then DDP's parameter-gradient average
recovers the single-process global objective. The scalar loss is not multiplied
by world size: doing that with detached remote tensors over-scaled trainable loss
parameters even though the encoder's DDP average canceled the embedding scale.
Trainable losses must participate in DDP synchronization with the encoder or
through their own DDP wrapper.

The gather uses
`torch.distributed._functional_collectives.all_gather_single`, the replacement
recommended by the deprecated `torch.distributed.nn.functional.all_gather`.
The module is private to PyTorch and may move in a later release; there is no
handwritten autograd collective or fallback. Batches
are padded for the collective and sliced back to their actual lengths; uneven
and empty ranks retain their gradient connection. All ranks must enter the
same operations in the same order and use consistent optional arguments,
dimensions, dtypes, and gradient requirements.

Every check that can differ between ranks runs locally before any collective:
embedding and label shapes and devices, reference inputs, index dtypes, and the
enqueue mask. One fixed-width metadata collective then carries each rank's
argument flags, its validation outcome, and the size, width, dtypes, and
gradient requirement of its query and reference inputs, and every rank applies
the same decision to the gathered rows: mismatched flags, an invalid rank, or a
mismatched schema raise on all ranks together, the invalid rank with its own
exception. A rank that raises between collectives leaves its peers blocked
until the process-group timeout, which previously happened for labels on the
wrong device. After the exchange, payload collectives see only inputs that
every rank has accepted.

Rank ordering also keeps every memory queue identical. Queue masks use the same
lengths and ordering as embeddings and labels. Replicas must start with the same
queue state and receive the same global inputs. The underlying memory loss still
owns capacity, detached references, wraparound, and index semantics.

Wrappers take `objective="global"` (the default) or `objective="local_anchor"`.
The two settings optimize different objectives with different gradients, so the
argument names the objective rather than a speed tradeoff. The local-anchor
objective: each
rank uses its local anchors and detached remote references, and returns world
size times its local loss. DDP therefore sums the per-rank surrogate gradients;
it does not recover full-batch reference gradients or a sample-weighted global
mean. Self-pair exclusion uses the rank's offset in the gathered reference set.
The miner wrapper removes self-pairs from the candidates it passes to the miner,
so a miner that selects one pair per anchor chooses among valid pairs instead of
losing a self-pair selection afterward. The loss wrapper passes the gathered
batch as `ref_emb` with a self-free pair tuple; losses that use `indices_tuple`
only as weights, such as `NCALoss`, still see each anchor's own row.
Explicit reference tensors describe a separate set, so matching row positions
are not self-pairs.

Memory losses support this mode through `CrossBatchMemory.forward_local_anchors`, which
separates the two roles that `forward` gives one tensor: anchors are the rank's rows with
their gradients, and the enqueued batch is the gathered one. `forward` is the case where
they are equal, so wraparound, masks, and capacity stay in one code path. Every rank
enqueues the gathered batch, not its own rows, because a rank's queue must hold other
ranks' embeddings to be useful, and identical queues keep checkpoints from any rank
equivalent. Self-pair exclusion selects the anchors' slots by the rank's offset. With a
mask, anchors are not enqueued, so there is nothing to exclude. Remote rows do not need
gradients because queue contents are always detached.

Inputs must be on the embeddings' device on every rank; the wrappers never
copy labels or masks across devices. Ordinary losses are
`BaseMetricLossFunction` or `MultipleLosses` instances (`types.MetricLoss`),
which share the query/reference forward signature. Candidate `indices_tuple`
arguments to the miner wrapper index one rank's batch and are rejected across
ranks.

The test process-group setup follows the
[PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
The parameter comparison helper was adapted from the
[PyTorch forum example](https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351).
The original process-group detection and gathering helpers were adapted from
[AllenAI's AllenNLP](https://github.com/allenai/allennlp) and
[John Giorgi's DeCLUTR](https://github.com/JohnGiorgi/DeCLUTR). Their attribution
is retained here as those helpers are replaced with PyTorch collectives.
