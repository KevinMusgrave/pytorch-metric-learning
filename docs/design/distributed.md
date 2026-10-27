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

CPU regression tests compare values, query/reference gradients, mined pairs, and
memory queue contents through wraparound for these single-process modes. They
exercise a real one-rank Gloo group and explicitly skip CUDA collectives. A
requested CUDA distributed run must fail when fewer than
two CUDA devices are visible. Returning successfully from an empty world-size
loop cannot establish distributed correctness.

GPU acceptance requires one host with two visible devices, a successful
two-rank NCCL collective, and executed distributed tests with their counts and
skip reasons recorded. These tests create their own worker processes; they must
not be wrapped in `torchrun`. Four-device cases can extend coverage later.

The test process-group setup follows the
[PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
The parameter comparison helper was adapted from the
[PyTorch forum example](https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351).
