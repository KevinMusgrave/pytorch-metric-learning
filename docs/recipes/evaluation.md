# Retrieval evaluation

Use this executable recipe from the repository root with the frozen development
environment. It extracts embeddings with an ordinary PyTorch loop, then passes
explicit query and reference tensors to
[`AccuracyCalculator`](../accuracy_calculation.md). It uses synthetic tensors,
completes in under a second on CPU, and performs no downloads. The Python file
beside this page is the canonical executable source; it is not an installed
library API.

| Task                                                                         | Executable source              | Command                                                       |
| ---------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------- |
| Extract embeddings and evaluate split, two-stream, or hierarchical retrieval | [evaluation.py](evaluation.py) | `uv run --frozen --no-sync python -m docs.recipes.evaluation` |

Verify the contracts with the same fixtures the removed testers used:

```sh
uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest tests/recipes/test_evaluation.py
```

The command prints `precision_at_1` and `r_precision` for one query split against
itself and against a separate gallery. The tests assert exact metric values for
each composition below.

## Extract embeddings

`extract_embeddings(model, loader)` iterates a `DataLoader` yielding
`(data, labels)`, concatenates `model(data)` in loader order, and L2-normalizes
each row. It runs under `torch.inference_mode()`, so outputs carry no autograd
graph. The caller chooses `model.eval()` or training mode, the device, the
dtype, any trunk/embedder composition, and any dimensionality reduction. The
function changes none of them. Drop the normalization when the selected
distance should see raw embeddings.

Labels are used as returned by the loader. Map strings or sparse identifiers to
integers before evaluation, for example with
`torch.unique(labels, return_inverse=True)` over the full label set.

## Compose query and reference splits

`split_accuracy(calculator, query, references, query_in_reference)` takes
`(embeddings, labels)` pairs. When `query_in_reference` is true, it places the
query first in the reference tensor and passes `ref_includes_query=True`, so
each query's own row is excluded from its neighbors. Additional reference splits
follow in the given order. When it is false, the reference is only the given
splits and no row is treated as a self-match.

| Query | References | Query in reference | Meaning                                           |
| ----- | ---------- | ------------------ | ------------------------------------------------- |
| `val` | none       | yes                | Leave-one-out retrieval within `val`              |
| `val` | `train`    | no                 | Retrieve `val` queries from a `train` gallery     |
| `val` | `train`    | yes                | Retrieve from `val ∪ train`, excluding each query |

Equal shapes or equal labels never imply self-matches. Pass the flag that
describes how the reference was assembled.

## Two-stream evaluation

`two_stream_accuracy(calculator, anchors, references, labels)` evaluates
anchors against a separate reference stream whose row `i` shares label `i` with
anchor `i`. The reference never includes the anchors, so no row is removed. This
replaces `GlobalTwoStreamEmbeddingSpaceTester`, which concatenated both streams
and then split them in half.

## Parent-conditioned hierarchical retrieval

`parent_conditioned_accuracy` takes labels shaped `(N, L)`, ordered from the
finest level to the coarsest. For each level `l < L - 1`, it evaluates level-`l`
labels within each parent group defined by column `l + 1`, considering only
references with that parent. The result for a level is the unweighted mean over
parent groups present in the query, under the key `<metric>_level<l>`. With more
than one evaluated level, `AVERAGE_<metric>` is the unweighted mean over levels.
For example, a cat query is ranked only against other animals, and a truck query
only against other vehicles. This replaces `WithSameParentLabelTester`.

For global, unconditioned evaluation of one label level, pass that column to
`split_accuracy` directly.

## Published benchmark splits

The library no longer downloads or parses datasets. The splits below are the
conventional metric-learning protocols the removed adapters implemented.
Applications load images with their own data pipeline and must respect each
dataset's license and terms.

| Dataset                                                                                                                                          | Training split                  | Test split                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- | ------------------------------ |
| [CUB-200-2011](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf), Wah et al.                                                         | Classes 1–100, 5,864 images     | Classes 101–200, 5,924 images  |
| Cars196, Krause et al., _3D Object Representations for Fine-Grained Categorization_                                                              | Classes 1–98, 8,054 images      | Classes 99–196, 8,131 images   |
| [Stanford Online Products](https://arxiv.org/abs/1511.06452), Oh Song et al.                                                                     | `Ebay_train.txt`, 59,551 images | `Ebay_test.txt`, 60,502 images |
| [iNaturalist 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.pdf), Van Horn et al. | 325,846 images                  | 136,093 images                 |

The iNaturalist retrieval split is the `Inaturalist_{train,test}_set1.txt`
protocol published with [Smooth-AP](https://github.com/Andrew-Brown1/Smooth_AP#data)
by Brown et al. Train and test classes are disjoint in every row, so evaluate the
test split against itself with `query_in_reference=True`.
