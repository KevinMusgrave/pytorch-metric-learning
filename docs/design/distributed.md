# Distributed wrappers

Distributed wrappers gather reference embeddings and labels across ranks while
preserving the gradients and local-anchor semantics required by the wrapped
loss or miner. They depend on numerical components and process-group state,
not on the training orchestration classes.

Without a process group, or with an initialized one-rank group, wrappers delegate
to the wrapped component without gathering or scaling. Memory losses receive
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

The local-anchor mode (`efficient=True`) retains the distinct objective: each
rank uses its local anchors and detached remote references, and returns world
size times its local loss. DDP therefore sums the per-rank surrogate gradients;
it does not recover full-batch reference gradients or a sample-weighted global
mean. Self-pair exclusion uses the rank's offset in the gathered reference set.
Explicit reference tensors describe a separate set, so matching row positions
are not self-pairs. Memory losses do not support this mode.

CPU regression tests use real one-rank and two-rank Gloo groups. Independent
contrastive and normalized-softmax formulas check values and gradients, including
an actual DDP model with a trainable encoder and loss. They cover uneven and
empty ranks, explicit references, supplied mined indices, local-anchor behavior,
queue wraparound with masks, and collective argument failures. A requested CUDA
run executes the same reference cases with NCCL and must fail when fewer than
two CUDA devices are visible.

GPU acceptance requires one host with two visible devices, a successful
two-rank NCCL collective, and executed distributed tests with their counts and
skip reasons recorded. These tests create their own worker processes; they must
not be wrapped in `torchrun`. Four-device cases can extend coverage later.

The test process-group setup follows the
[PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
The parameter comparison helper was adapted from the
[PyTorch forum example](https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351).
The original process-group detection and gathering helpers were adapted from
[AllenAI's AllenNLP](https://github.com/allenai/allennlp) and
[John Giorgi's DeCLUTR](https://github.com/JohnGiorgi/DeCLUTR). Their attribution
is retained here as those helpers are replaced with PyTorch collectives.
