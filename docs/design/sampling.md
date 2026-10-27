# Metric-learning sampling

Samplers control label and tuple structure while PyTorch owns data loading,
optimizers, and training. This module depends only on the numerical core;
sampling must not import evaluation or model traversal.

Samplers use PyTorch's random number generators: an explicit
`torch.Generator`, or the global generator when none is given, so seeding is
the same as for other PyTorch randomness. The library has no global random
state of its own. Each iterator is an independent draw. Sampling balances
labels within each batch and samples with replacement across batches; it does
not promise epoch coverage, which several issue reports expected. `len()`
equals the number of items an iterator yields. Index samplers go to
`DataLoader(sampler=...)`; the hierarchical sampler yields index lists for
`batch_sampler=` and computes its length without materializing batches.

An M-per-class sampler preserves its batch label counts and may sample with
replacement when a class has fewer than M members. Hierarchical sampling uses
the provided label columns to balance the requested levels. These are behavior
contracts, independent of a training framework. Fixed triplets preserve the
anchor/positive/negative label relation.

Mining-weighted sampling belongs to the agent-facing recipe in
`docs/recipes/mining_weights.py`: callers compute embeddings, the numerical
tuple utility derives weights, and PyTorch's WeightedRandomSampler draws
dataset indices. The library has no model-aware sampler or tester dependency.
