# Executable metric-learning recipes

The executable recipes are ordinary PyTorch application code beside their
canonical documentation, outside the distributed package. They show
algorithm-specific composition without introducing a training framework. The library does not
own optimizers, model modes, data loading, schedules, checkpointing, or logging.
Each recipe runs on small synthetic tensors without network access. There is no
standalone example gallery, tutorial collection, or Jupyter notebook surface.
Each topic states inputs, expected results, supported composition, and an exact
verification command for an agent to retrieve and execute.

Basic, classifier-assisted, learnable-proxy, and two-stream loops make the
optimized parameters explicit. A proxy optimizer includes the loss parameters.
Separate reference tensors remain attached to autograd. Scalar conversion in
example history is an explicit diagnostic, not part of the library hot path.

Cascaded mining filters each embedding slice using the previous stage's pairs,
passed to `HDCMiner` as its `indices_tuple`; the miner keeps no state that
could leak into later batches. Stage losses are summed with equal weights in the
example; applications can choose weights and classifier heads explicitly.

The adversarial example preserves the historical generator's reverse-triplet,
hardness, and regularization formulas. It makes alternating gradient ownership
explicit: generator updates use detached trunk features and a frozen embedder;
metric updates use detached generated negatives and optimize the trunk and
embedder. The generator's adversarial hinge averages only positive violations,
while hardness and regularization are mean squared errors over normalized
feature coordinates. These conventions describe the extracted implementation,
not a claim to reproduce a paper's training results. The synthetic example uses
linear/ReLU modules; applications with stateful layers must choose their model
modes deliberately as well as controlling gradients.

Mining-weighted sampling computes embeddings in caller code. Unique sampled
dataset indices map tuple-frequency weights back to the original dataset;
unselected rows receive zero. `convert_to_weights` gives zero weights for an
empty mined tuple, which a sampler cannot draw from, so the recipe passes `None`
instead and samples the selected subset uniformly. PyTorch's sampler owns
replacement and RNG state. Memory warmup trains against the current batch only,
then starts filling the detached queue at the caller-selected step.

The evaluation recipe extracts embeddings in caller code and composes
explicit query/reference sets for `AccuracyCalculator`. Split composition places
a query first when it belongs to the reference; two-stream evaluation never
removes self-matches; parent-conditioned metrics are unweighted means over
parent groups, then over levels. Published dataset splits are documentation, not code.

Reference tests verify objective values and gradients, candidate-set nesting,
gradient ownership, parameter updates, subset mapping, and queue progression.
Runnable commands and paper citations belong in `docs/recipes/training.md`.
