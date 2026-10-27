# Distributed wrappers

Distributed wrappers gather reference embeddings and labels across ranks while
preserving the gradients and local-anchor semantics required by the wrapped
loss or miner. They depend on numerical components and process-group state,
not on the training orchestration classes.

A CPU run verifies the loss wrapper’s single-process fallback and explicitly skips
CUDA collectives. A requested CUDA distributed run must fail when fewer than
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
