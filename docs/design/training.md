# Training and sampling

Trainers compose user-provided models, optimizers, losses, miners, and evaluation
hooks. Samplers control the label structure of batches; they do not own losses
or optimizer updates. Sampling may depend on evaluation when it derives weights
from embedding quality, which is why this boundary depends on both numerics and
evaluation.

Routine training tests use small synthetic tensor datasets and temporary output
directories. They must exercise optimization, saved models, recorded losses,
and evaluation hooks without cloning a model repository or downloading a public
dataset. Dataset size and model complexity are not necessary to verify those
contracts.

An M-per-class sampler preserves its batch label counts and may sample with
replacement when a class has fewer than M members. Hierarchical sampling uses
the provided label columns to balance the requested levels. These are behavior
contracts, independent of a historical training script or framework version.
