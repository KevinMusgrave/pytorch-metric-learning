# Training composition

Use these executable recipes from the repository root with the frozen
development environment. They use synthetic tensors, complete in seconds on
CPU, and perform no downloads. The Python files beside this page are the
canonical executable source; they are not installed as library APIs.

For installation and the public component index, start at [the documentation
index](../index.md). Retrieval evaluation has its own
[evaluation recipe](evaluation.md). The application owns model modes, device/dtype placement,
optimizers, data loading, logging, and checkpoints.

## Choose a recipe

| Task                                                       | Executable source                                                | Command                                                                        |
| ---------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Metric, classifier-assisted, proxy, or two-stream training | [training.py](training.py)                                       | `uv run --frozen --no-sync python -m docs.recipes.training`                    |
| Progressive mining over embedding slices                   | [cascaded_mining.py](cascaded_mining.py)                         | `uv run --frozen --no-sync python -m docs.recipes.cascaded_mining`             |
| Alternating adversarial metric learning                    | [adversarial_metric_learning.py](adversarial_metric_learning.py) | `uv run --frozen --no-sync python -m docs.recipes.adversarial_metric_learning` |
| Mining-weighted dataset sampling                           | [mining_weights.py](mining_weights.py)                           | `uv run --frozen --no-sync python -m docs.recipes.mining_weights`              |
| Caller-controlled memory warmup                            | [memory_warmup.py](memory_warmup.py)                             | `uv run --frozen --no-sync python -m docs.recipes.memory_warmup`               |

The commands import the installed package, so install the checkout as described
in the [development guide](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/DEVELOPMENT.md#set-up-a-checkout),
or prefix them with `PYTHONPATH=src`. Verify the contracts and optimization
behavior together:

```sh
uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest tests/recipes/test_training.py
```

## Metric, classifier, proxy, and two-stream inputs

`train_metric` takes a model, floating-point data, and one-dimensional integer
class labels. The model returns `(N, D)` embeddings. Its pair miner selects
positive distances above zero and negative distances below 1.4 after L2
normalization; the contrastive loss averages active positive and negative
violations separately, then sums them. The optional classifier maps embeddings
to logits for contiguous labels `0..C-1`; its cross-entropy has weight 1.

`train_proxy` uses normalized-softmax loss with temperature 0.1. Its optimizer
owns both model parameters and the loss's class-proxy matrix. Omitting that
matrix from the optimizer leaves the proxies fixed.

`train_two_stream` takes distinct query/reference models, data, and labels.
The streams can have different lengths, share class labels, and both receive
gradients. Equal positions are not treated as self-matches merely because the
streams have equal lengths. The recipe uses distinct models; if sharing one
model, include its parameters only once in the optimizer.

The training command reports initial and final objectives for four integrations.
The fixed synthetic fixtures decrease their objectives and update every intended
parameter owner. This is an integration check, not evidence of improved training
quality on real data. Inputs and modules must already share the desired device
and dtype; these functions do not move or change the mode of caller models.

## Cascaded mining contract

The recipe follows the composition from Yuan et al.,
[Hard-Aware Deeply Cascaded Embedding](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf).
Split `(N, D)` embeddings into three slices with sizes summing to `D`. Each
[HDCMiner](../miners.md#hdcminer) stage receives the previous stage's pairs as
its `indices_tuple` candidates and retains the hardest fraction of the positive
and negative pairs; later stages cannot introduce pairs rejected earlier. The
executable uses fractions `(1, 0.5, 0.5)` and sums three contrastive losses. The
output reports a finite loss and positive-pair counts `24, 12, 6` on the
supplied fixture.

The recipe applies the paper's progressive candidate filtering to arbitrary
embedding slices and losses; it is not a reproduction of the paper's
architecture or results. The number of miners must equal the number of slices,
and the sizes must sum to the embedding dimension; `torch.split` and
`zip(..., strict=True)` raise on mismatches.

## Adversarial objectives and gradient ownership

The recipe derives from Duan et al.,
[Deep Adversarial Metric Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf),
and computes three objectives. Let `a`, `p`, `n` be trunk features and `s = G([a,p,n])` the generated negative.
Let `u(x)` L2-normalize a feature and `e(x)` L2-normalize its embedder output.

- Adversarial: positive-violation mean of
  `max(||e(a)-e(s)||₂ - ||e(a)-e(p)||₂ - 0.2, 0)`; zero when no violation is active.
- Hardness: mean squared coordinate difference between `u(s)` and `u(a)`.
- Regularization: mean squared coordinate difference between `u(s)` and `u(n)`.

The generator step gives these three terms equal weight. It detaches trunk
features and freezes embedder parameters while preserving gradients through
the embedder to generated features. The metric step detaches generated
negatives and optimizes trunk/embedder parameters with authentic and synthetic
triplet-margin losses, each with margin 0.2 and weight 1. Generate nonempty
valid triplets from batches containing repeated labels and multiple classes.

The command alternates ten generator/metric updates and prints finite final
objectives. Tests check analytical values, gradients, and that each optimizer
changes only its intended parameters across repeated alternating steps. The
linear/ReLU smoke architecture and equal weights are explicit recipe settings,
not paper-reproduction settings. For stateful layers, choose train/eval modes
deliberately as well as controlling parameter gradients. Schedule warmup or
classifier terms explicitly in application code if needed.

## Mining-weighted sampling contract

Compute embeddings in caller code, mine tuples, then map per-subset occurrence
weights to the full dataset before using `torch.utils.data.WeightedRandomSampler`.
Weights are occurrence count divided by the maximum count. Subset indices must
be unique dataset indices and align with the subset labels and mined embeddings.
Unselected dataset rows receive zero weight; empty mined tuples give uniform
weights within the selected subset. This is sampling with replacement, not an
epoch-coverage guarantee. The recipe uses a local seeded PyTorch generator and
reports four batches of eight labels. The tests verify exact subset mapping and
that sampling never selects a zero-weight row.

## Memory warmup contract

The recipe trains on the current batch for two steps without touching the queue,
then enqueues detached embeddings for four steps through `CrossBatchMemory`.
Its batch size is 12, embedding size 4, and memory capacity 32, so the final
queue is full with cursor 16. The queue participates only after warmup. No
internal epoch counter or training hook decides when it starts.

Wang et al., [Cross-Batch Memory for Embedding Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf),
motivate delayed memory use by slow feature drift. Two steps and capacity 32
only make the integration executable quickly; they do not claim the paper's
training schedule or memory size. Choose those from the intended experiment.
See [CrossBatchMemory](../losses/cross_batch_memory.md#crossbatchmemory) for supported losses,
enqueue masks, and queue ownership.
