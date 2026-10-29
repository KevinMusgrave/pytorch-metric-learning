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

The gather uses the current pinned PyTorch's
`torch.distributed._functional_collectives.all_gather_single`, the replacement
recommended by its deprecated `torch.distributed.nn.functional.all_gather`.
The module is private to PyTorch, so a PyTorch upgrade must re-run the
distributed tests and, if the function moves, switch to its public successor.
There is no handwritten autograd collective or old-version fallback. Batches
are padded for the collective and sliced back to their actual lengths; uneven
and empty ranks retain their gradient connection. Small metadata collectives
establish sizes and reject mismatched schemas before payload communication.
All ranks must enter the same operations in the same order and use consistent
optional arguments, dimensions, dtypes, and gradient requirements.

Rank ordering also keeps every memory queue identical. Queue masks use the same
lengths and ordering as embeddings and labels. Replicas must start with the same
queue state and receive the same global inputs. The underlying memory loss still
owns capacity, detached references, wraparound, and index semantics.

Wrappers take `objective="global"` (the default) or `objective="local_anchor"`.
A boolean `efficient` flag described a speed tradeoff, but the two settings
optimize different objectives with different gradients, so the argument names
the objective. The local-anchor objective: each
rank uses its local anchors and detached remote references, and returns world
size times its local loss. DDP therefore sums the per-rank surrogate gradients;
it does not recover full-batch reference gradients or a sample-weighted global
mean. Self-pair exclusion uses the rank's offset in the gathered reference set.
Explicit reference tensors describe a separate set, so matching row positions
are not self-pairs. Memory losses do not support this mode.

Inputs must be on the embeddings' device on every rank; the wrappers never
copy labels or masks across devices. Candidate `indices_tuple` arguments to the
miner wrapper index one rank's batch and are rejected across ranks.

The test process-group setup follows the
[PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
The parameter comparison helper was adapted from the
[PyTorch forum example](https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351).
The original process-group detection and gathering helpers were adapted from
[AllenAI's AllenNLP](https://github.com/allenai/allennlp) and
[John Giorgi's DeCLUTR](https://github.com/JohnGiorgi/DeCLUTR). Their attribution
is retained here as those helpers are replaced with PyTorch collectives.
