# Metric-learning sampling

Samplers control label and tuple structure while PyTorch owns data loading,
optimizers, and training. This module depends only on the numerical core;
sampling must not import evaluation or model traversal.

An M-per-class sampler preserves its batch label counts and may sample with
replacement when a class has fewer than M members. Hierarchical sampling uses
the provided label columns to balance the requested levels. These are behavior
contracts, independent of a training framework. Fixed triplets preserve the
anchor/positive/negative label relation.

Mining-weighted sampling belongs to the agent-facing recipe in
`docs/recipes/mining_weights.py`: callers compute embeddings, the numerical
tuple utility derives weights, and PyTorch's WeightedRandomSampler draws
dataset indices. The library has no model-aware sampler or tester dependency.
