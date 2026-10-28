# Modernization component decisions

These record the scope decisions made at the start of version 3 modernization.
The starting inventory below describes the old package, including components
since removed. Current implementation rationale remains in `docs/design/`;
the [release notes](../release-notes-3.md) describe the implemented public API.

## Inventory and target surface

The starting inventory is the public imports at `5a2d9d7`. Counts include
historically exported base classes and mixins; they are not a target API size.

| Component                     | Starting surface                                                                                                                     | Decision and reason                                                                                                                                                                                                   | Evidence to preserve                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Losses                        | 43 exports in `losses`                                                                                                               | Retain mathematical objectives and useful wrappers; simplify bases and composition. Remove regularizer mixins from the public API after replacing their behavior explicitly.                                          | `tests/losses`, paper references in `docs/losses.md`, independent values and gradients.                        |
| Miners                        | 11 exports in `miners`                                                                                                               | Retain metric-specific tuple selection. Clarify candidate sets, margins, sampling, and index conventions.                                                                                                             | `tests/miners`, paper formulas, positive/negative label validity and sampling distributions.                   |
| Distances                     | 6 exports in `distances`                                                                                                             | Retain tensor distances, similarity orientation, and bounded-memory evaluation. Make normalization explicit in contracts.                                                                                             | `tests/distances`, normalization and dtype behavior, forward/backward benchmarks.                              |
| Reducers                      | 10 exports in `reducers`                                                                                                             | Retain meaningful averaging/filtering/weighting choices; simplify dictionary dispatch and composition. Internal containers need not remain public.                                                                    | `tests/reducers`, empty-set behavior, per-anchor versus per-tuple weighting.                                   |
| Regularizers                  | 6 exports in `regularizers`                                                                                                          | Retain numerical penalties and explicit weighting; remove dependency on MRO or name discovery.                                                                                                                        | `tests/regularizers`, weight and embedding gradients.                                                          |
| Memory and self-supervision   | `CrossBatchMemory`, `SelfSupervisedLoss`, `MultipleLosses`                                                                           | Retain composition that changes metric-learning behavior. Define queue ownership, gradient detachment, and loss weighting.                                                                                            | Existing wrapper tests; add wraparound, multi-forward/backward, and distributed queue references.              |
| Distributed support           | Loss/miner wrappers in `utils.distributed`                                                                                           | Retain, with documented gather ordering, gradient/scaling semantics, and supported combinations. Repair fallback and queue behavior before API redesign.                                                              | CPU references and actual two-GPU NCCL acceptance.                                                             |
| Label samplers                | `MPerClassSampler`, `HierarchicalSampler`, `FixedSetOfTriplets`                                                                      | Retain domain-specific batch/tuple construction. Use the correct PyTorch sampler protocol and explicit reproducibility controls. Sampling with replacement is not an epoch-coverage guarantee.                        | `tests/samplers`, batch label counts, valid indices, seeded behavior.                                          |
| Mining-weighted sampling      | `TuplesToWeightsSampler`                                                                                                             | Move model/dataset traversal to an example. Retain tuple-frequency weighting as a tensor utility usable with PyTorch `WeightedRandomSampler`.                                                                         | Weight formula and subset-index mapping from current tests.                                                    |
| Retrieval metrics             | `AccuracyCalculator` and metric functions                                                                                            | Retain mathematical metrics. Preserve label dtypes, explicit query/reference identity, self exclusion, custom label comparisons, and per-class aggregation.                                                           | `tests/utils/test_calculate_accuracies.py` and retrieval references.                                           |
| Retrieval backends            | `CustomKNN`, `FaissKNN`, `FaissKMeans`, matching helpers                                                                             | Retain PyTorch nearest-neighbor and threshold matching operations. Make Faiss and clustering dependencies optional and requested explicitly. Do not make core loss imports initialize optional backends.              | KNN/metric tests; distance orientation, ties, empty inputs, explicit missing-backend errors.                   |
| Model inference orchestration | `InferenceModel`, PCA helper                                                                                                         | Remove automatic model mode/device changes, dataset traversal, and generic PCA wrapping. Show ordinary PyTorch extraction followed by a retrieval primitive. Faiss index persistence stays with the optional backend. | Retain independent retrieval tests; replace model wrappers with an executed extraction/retrieval example.      |
| Testers                       | `BaseTester`, `GlobalEmbeddingSpaceTester`, `GlobalTwoStreamEmbeddingSpaceTester`, `WithSameParentLabelTester`                       | Remove the tester hierarchy. Move embedding extraction and split orchestration to examples; preserve two-stream and parent-conditioned metric semantics in recipes.                                                   | Extract numerical fixtures from `tests/testers` before deleting orchestration tests.                           |
| Trainers                      | `BaseTrainer`, `MetricLossOnly`, `TrainWithClassifier`, `TwoStreamMetricLoss`, `CascadedEmbeddings`, `DeepAdversarialMetricLearning` | Remove the framework after extracting the recipes below. Optimizers, schedules, checkpointing, freezing, and logging belong to applications.                                                                          | Plain PyTorch examples with observable optimization; mathematical pieces from the algorithm-specific trainers. |
| Logging                       | `logging_presets`, record-keeper hooks, TensorBoard integration                                                                      | Remove integration and its dependencies. Retain useful opt-in numerical statistics without default host synchronization or a logger requirement.                                                                      | Statistics tests for retained modules; no replacement logging framework.                                       |
| Datasets                      | `BaseDataset`, `CUB`, `Cars196`, `INaturalist2018`, `StanfordOnlineProducts`                                                         | Remove adapters/downloaders from the package. Their primary behavior is metadata parsing and data acquisition. Keep source citations and published split descriptions in an evaluation recipe.                        | Cite dataset authors and canonical splits; maintained examples use synthetic tensors and need no downloads.    |
| Shared utilities              | `common_functions`, tuple utilities, record modules, key checking, loss tracking                                                     | Retain only tensor/index/statistics helpers used by the final library. Remove trainer key schemas, model checkpoint management, URL retrieval, SQLite conversion, and orphan helpers with their callers.              | Numerical utility tests; package import/dependency checks after pruning.                                       |

Retained numerical objectives are not removed merely because a linter reports
no local caller. Public exports will be explicit. A proven defect requires a
reference-backed fix or a documented unsupported combination, not a silent
change to the objective.

## Recipes to extract before deletion

These are small agent-facing executable recipes colocated with the canonical
contracts, not a standalone examples gallery or a tutorial collection. Each
states its inputs, supported composition, expected results, and verification
command. Remove every Jupyter notebook; keep only recipes that preserve useful
metric-specific behavior or make the public API verifiable.

- **Basic and classifier-assisted training:** a normal PyTorch loop with a loss,
  optional miner, optional classifier term, and optimizer steps. A proxy example
  must optimize the loss's own parameters as well as the embedding model.
- **Two-stream learning/evaluation:** anchors and references are distinct tensor
  sets. Preserve reference gradients where appropriate and do not infer self
  matches from equal labels or equal lengths.
- **Cascaded mining:** preserve progressive mining over embedding slices and the
  `HDCMiner` candidate filtering used by the historical trainer. Cite Yuan et al.,
  [Hard-Aware Deeply Cascaded Embedding](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf).
  Explain that the old trainer generalized the paper; do not label a synthetic
  smoke example a reproduction of its training results.
- **Adversarial metric learning:** preserve the generator's adversarial,
  hardness, and regularization objectives and alternating optimization, with
  explicit gradient ownership. Cite Duan et al.,
  [Deep Adversarial Metric Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf).
  Validate the objective pieces before deleting the trainer implementation.
- **Hierarchical retrieval:** compute metrics within each parent-label subset
  and preserve the documented aggregation across parents and levels.
- **Mining-weighted sampling:** compute embeddings in caller code, derive
  example weights from mined tuples, and pass them to PyTorch's sampler.
- **Memory warmup:** demonstrate caller-controlled warmup and queue filling,
  including how settings differ from the cross-batch-memory paper.

## Dependencies and breaking changes

The numerical core must not depend on evaluation, training, datasets, Faiss,
scikit-learn, TensorBoard, or record-keeper. Sampling must not depend on testers.
Evaluation may depend on the numerical core and explicit optional retrieval or
clustering backends. Distributed wrappers depend on the numerical core and
PyTorch process groups.

Remove `record-keeper` and `tensorboard` when their integration disappears.
Move `scikit-learn` to an optional clustering extra if its retained metrics still
require it. Keep Faiss optional. Remove Pillow, torchvision, SciPy, and tqdm
where no retained code or maintained example needs them. NumPy remains a
declared core dependency: it is standard alongside PyTorch, and removing it
would not justify the churn. Update the exact lock and security assessments together,
using the pinned dependency-review workflow.

The next major release removes trainers, testers, `InferenceModel`, dataset
downloaders, logging hooks, and their associated extras. No aliases or migration
shims are planned. Document replacement recipes and retained tensor contracts.
Normalization, reduction, tuple weighting, and paper defaults must be stated
explicitly; changing any of them requires correctness evidence and release notes.

## Open-work priorities

Read-only GitHub inventory on 2026-10-27 UTC found 72 open issues and 3 open pull
requests. The following reports inform this plan; they are not assertions that
every reported defect has been reproduced. No issue or PR was modified.

| Priority                        | Reports                                                                                                                                                                                                                                                                                                                                                                                                                                          | Treatment                                                                                                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness: numerical sampling | [PR 764](https://github.com/KevinMusgrave/pytorch-metric-learning/pull/764), by **kocean9-freedom**                                                                                                                                                                                                                                                                                                                                              | Review the eligible-negative, per-anchor stabilization fix against independent probability references and land it through the original PR.                            |
| Correctness: label identity     | [PR 763](https://github.com/KevinMusgrave/pytorch-metric-learning/pull/763), by **akshitkhokhani**                                                                                                                                                                                                                                                                                                                                               | Review large-integer label relabeling invariance and exact self-match identity and land the fix through the original PR.                                              |
| Correctness: distributed memory | [639](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/639), [748](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/748), [448](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/448)                                                                                                                                                                                                               | Test deterministic queue ordering, queue reuse, and gradients. Define local-anchor memory support explicitly.                                                         |
| Correctness: stateful backward  | [731](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/731)                                                                                                                                                                                                                                                                                                                                                                       | Reproduce DynamicSoftMarginLoss's reported in-place version failure with multiple forwards before backward.                                                           |
| Correctness: mined pairs        | [723](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/723), [691](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/691)                                                                                                                                                                                                                                                                                           | Verify label validity and epsilon semantics against the paper, including MPS when supported and available.                                                            |
| Contracts: samplers             | [749](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/749), [612](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/612), [684](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/684), [725](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/725)                                                                                                                                   | Fix sampler/batch-sampler integration; distinguish class balance from complete epoch coverage. Retire the trainer-specific dispatch.                                  |
| Contracts: retrieval            | [730](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/730), [737](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/737), [738](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/738), [602](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/602), [430](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/430)                                                       | Use explicit backend selection, valid device handling, clear invalid-input errors, and consistent distance semantics.                                                 |
| Contracts: defaults and tuples  | [670](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/670), [496](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/496), [202](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/202), [466](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/466), [614](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/614)                                                       | Specify normalization, query/reference identity, tuple conversion/weighting, and the supported memory API before simplifying it.                                      |
| Class design                    | [437](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/437), [587](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/587)                                                                                                                                                                                                                                                                                           | Replace regularizer name discovery and simplify reducers using representative pair/proxy prototypes.                                                                  |
| Performance and customization   | [211](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/211), [586](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/586), [697](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/697), [193](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/193), [194](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/194)                                                       | Benchmark duplicate distance work and large candidate sets before adding caching or new interfaces.                                                                   |
| Paper fidelity and examples     | [567](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/567), [478](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/478), [633](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/633), [641](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/641)                                                                                                                                   | Add paper settings, memory warmup, self-supervision examples, and contribution guidance with citations.                                                               |
| Retired orchestration           | [753](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/753), [717](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/717), [722](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/722), [513](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/513)                                                                                                                                   | Replace obsolete notebooks/checkpoint helpers and do not expand the dataset framework. Explain disposition when issue maintenance is authorized.                      |
| New algorithms                  | [PR 452](https://github.com/KevinMusgrave/pytorch-metric-learning/pull/452), by **chingisooinar**, and algorithm requests such as [658](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/658), [557](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/557), [675](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/675), [103](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/103) | Defer new algorithm adoption until retained contracts are stable. Preserve the proposals and author credit; modernization does not promise every requested algorithm. |

## Attribution through the cutover

Keep paper citations with algorithm documentation and examples. Record adapted
implementation provenance and required upstream license text in distributable
third-party notices before removing corresponding source comments. Keep existing
contributor acknowledgments when condensing the README; move them to a dedicated
credits document if necessary. Land contributed fixes through their original PRs, preserving authorship and
review history. When revisions are needed and maintainer edits are enabled,
push those revisions to the contributor's branch before merging the PR. A ready
PR needs no extra push. Do not absorb its patch into an unrelated modernization
commit. Both PR 763 and PR 764 currently permit maintainer edits and target
`master`. Defer their integration until the modernization branch has merged into
`master`; then update each original PR against that baseline, adapt it to the
retained API, verify it, and merge it separately. Keep their reported defects
visible during modernization without absorbing their patches. Publishing and
merging remain separately authorized operations.
A rewrite does not erase the provenance of retained algorithms or adapted code.
