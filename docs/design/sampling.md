# Metric-learning sampling

Samplers control label and tuple structure while PyTorch owns data loading,
optimizers, and training. This module depends only on the numerical core;
sampling must not import evaluation or model traversal.

Samplers use PyTorch's random number generators: an explicit
`torch.Generator`, or the global generator when none is given, so seeding is
the same as for other PyTorch randomness. The library has no global random
state of its own. Each iterator is an independent draw. `len()` equals the
number of items an iterator yields. Samplers whose guarantee is a batch's
composition yield index lists for `batch_sampler=`: an index sampler would leave
that guarantee to a matching `DataLoader` batch size, and a mismatch silently
breaks it. Hierarchical sampling computes its length without materializing
batches.

An M-per-class sampler preserves its batch label counts in every batch and
repeats examples within a batch only when a class has fewer than M members.
Balance and coverage conflict: with unequal class sizes, equal class counts per
batch cannot visit every example once. Balance wins. Classes and each class's
examples rotate through shuffled passes, which keeps counts within one of each
other at every point in an iterator, so a default-length iterator can state a
provable result: every example at least once, and exactly once for equal class
sizes that divide evenly. Weighting classes by size would spend fewer batches
on small classes, but drops the equal class counts callers choose this sampler
for. Hierarchical sampling uses the provided label columns to balance the
requested levels and keeps independent draws. These are behavior contracts,
independent of a training framework. Fixed triplets preserve the
anchor/positive/negative label relation.
Their consecutive triples map directly to an `indices_tuple` through
`torch.arange(batch_size).reshape(-1, 3).unbind(1)` on the embeddings' device;
there is no miner class for recovering already supplied triplets. Fixed pairs
are a batch sampler because their guarantee is a batch's composition: an odd
batch size would split a pair and shift every later pair onto wrong partners.
Positive pairs need two different examples, because a pair of an example with
itself has zero distance and teaches nothing. The caller derives positive and
negative rows from batch labels, so no pair-specific miner or label-free dataset
format is added.

Fixed tuples draw ordered distinct class pairs with two bounded integer draws:
the first is uniform over all C classes, and the second is uniform over C - 1
positions with the first position skipped. Every ordered distinct pair has
probability 1 / (C * (C - 1)), independent of class size. Drawing in blocks of
at most 1024 avoids a full C-element permutation for each tuple without
materializing all class-pair choices at once. Member selection and tuple
ordering keep their existing rules; the random stream is reproducible with the
new algorithm but is not the same stream as the prior implementation.

Mining-weighted sampling belongs to the executable recipe in
`docs/recipes/mining_weights.py`: callers compute embeddings, the numerical
tuple utility derives weights, and PyTorch's WeightedRandomSampler draws
dataset indices. The library has no model-aware sampler or tester dependency.
