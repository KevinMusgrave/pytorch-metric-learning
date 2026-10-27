# Executable metric-learning recipes

The agent-facing recipes are ordinary PyTorch application code beside their
canonical documentation, outside the distributed package. They retain
algorithm-specific composition previously embedded in
trainers without introducing a new training framework. The library does not
own optimizers, model modes, data loading, schedules, checkpointing, or logging.
Each recipe runs on small synthetic tensors without network access. There is no
standalone example gallery, tutorial collection, or Jupyter notebook surface.
Each topic states inputs, expected results, supported composition, and an exact
verification command for an agent to retrieve and execute.

Basic, classifier-assisted, learnable-proxy, and two-stream loops make the
optimized parameters explicit. A proxy optimizer includes the loss parameters.
Separate reference tensors remain attached to autograd. Scalar conversion in
example history is an explicit diagnostic, not part of the library hot path.

Cascaded mining filters each embedding slice using the previous stage's pairs.
Reset the HDC miner's external candidates after each call so subsequent batches
do not inherit stale indices. Stage losses are summed with equal weights in the
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
unselected rows receive zero. An empty mined tuple follows the existing utility
contract of uniform weights over the selected subset. PyTorch's sampler owns
replacement and RNG state. Memory warmup trains against the current batch only,
then starts filling the detached queue at the caller-selected step.

Reference tests verify objective values and gradients, candidate-set nesting,
gradient ownership, parameter updates, subset mapping, and queue progression.
Runnable commands and paper citations belong in `docs/recipes/training.md`.
