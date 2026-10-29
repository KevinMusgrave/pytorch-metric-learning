# Migrating from version 2 to version 3

Search for the old import, class, method, or argument below. Names are relative
to `pytorch_metric_learning` unless another package is named. A removed class's
direct module import follows the same replacement as its package export.
Version 3 provides no compatibility aliases. See the [release notes](release-notes-3.md)
for the reasons behind the changes and corrected numerical behavior.

## Installation and shared arguments

| v2 symbol or call                                                            | v3 replacement                                                         | Note                                                                                                          |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Python 3.10 or 3.11                                                          | Python 3.12 or newer                                                   | Minimum tested runtimes are PyTorch 2.14.0 and NumPy 2.5.3.                                                   |
| `pytorch-metric-learning[with-hooks]`                                        | Application-owned logging; install `[faiss]` separately when needed    | record-keeper and TensorBoard integration were removed.                                                       |
| `pytorch-metric-learning[datasets]`                                          | Application dataset dependencies                                       | The library no longer downloads datasets.                                                                     |
| `pytorch-metric-learning[docs]`                                              | The frozen development environment                                     | See [development setup](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/DEVELOPMENT.md). |
| Implicit scikit-learn installation                                           | `pytorch-metric-learning[clustering]`                                  | Required for NMI and AMI, alongside an explicit clustering backend.                                           |
| `collect_stats=True` on losses, miners, distances, reducers, or regularizers | Compute diagnostics from returned tensors or use PyTorch forward hooks | Recorded statistic attributes were removed; see [statistics](release-notes-3.md#statistics-removed).          |
| Catching `AssertionError` for invalid arguments                              | Catch `TypeError` or `ValueError`                                      | Validation remains active under `python -O`.                                                                  |

## Trainers

| v2 symbol or call                                              | v3 replacement                                            | Note                                                                                         |
| -------------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `trainers.BaseTrainer`                                         | An ordinary PyTorch training loop                         | Applications own data loading, optimizers, schedules, freezing, and checkpoints.             |
| `trainers.MetricLossOnly`                                      | Compute embeddings, call the loss, then backward and step | [Metric training recipe](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs). |
| `trainers.TrainWithClassifier`                                 | Add classifier and metric objectives in caller code       | [Classifier recipe](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs).      |
| `trainers.TwoStreamMetricLoss`                                 | Pass explicit query and reference embeddings to the loss  | [Two-stream recipe](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs).      |
| `trainers.CascadedEmbeddings`                                  | Pass mined candidate pairs to successive `HDCMiner` calls | [Cascaded mining recipe](recipes/training.md#cascaded-mining-contract).                      |
| `trainers.DeepAdversarialMetricLearning`                       | Explicit metric, classifier, and adversarial objectives   | [Adversarial recipe](recipes/training.md#adversarial-objectives-and-gradient-ownership).     |
| `trainers.deep_adversarial_metric_learning.NegativeLpDistance` | Form the adversarial distance objective in caller code    | The adversarial recipe specifies signs and gradient ownership.                               |

## Testers

| v2 symbol or call                             | v3 replacement                                                               | Note                                                                                       |
| --------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `testers.BaseTester`                          | Extract embeddings, then call `utils.accuracy_calculator.AccuracyCalculator` | [Embedding extraction](recipes/evaluation.md#extract-embeddings).                          |
| `testers.GlobalEmbeddingSpaceTester`          | Explicit query/reference split composition                                   | [Split composition](recipes/evaluation.md#compose-query-and-reference-splits).             |
| `testers.GlobalTwoStreamEmbeddingSpaceTester` | Evaluate one stream against the other                                        | [Two-stream evaluation](recipes/evaluation.md#two-stream-evaluation).                      |
| `testers.WithSameParentLabelTester`           | Evaluate each parent-conditioned gallery explicitly                          | [Hierarchical retrieval](recipes/evaluation.md#parent-conditioned-hierarchical-retrieval). |

## Datasets

| v2 symbol or call                 | v3 replacement                                      | Note                                                                                           |
| --------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `datasets.BaseDataset`            | `torch.utils.data.Dataset`                          | Applications own downloads, transforms, labels, and split selection.                           |
| `datasets.Cars196`                | An application dataset for Cars196                  | Preserve the [published split and citation](recipes/evaluation.md#published-benchmark-splits). |
| `datasets.CUB`                    | An application dataset for CUB-200-2011             | Preserve the published split and citation.                                                     |
| `datasets.INaturalist2018`        | An application dataset for iNaturalist 2018         | Preserve the published split and citation.                                                     |
| `datasets.StanfordOnlineProducts` | An application dataset for Stanford Online Products | Preserve the published split and citation.                                                     |

## Utils

### Retrieval and accuracy calculation

| v2 symbol or call                                           | v3 replacement                                                | Note                                                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `utils.inference.InferenceModel`                            | Extract embeddings in caller code, then use `utils.retrieval` | [Evaluation recipe](recipes/evaluation.md); preprocessing and model mode belong to the application. |
| `InferenceModel.train_indexer`                              | `utils.retrieval.FaissKNN.train(embeddings)`                  | Extract the reference embeddings first; use `add`, `save`, and `load` on the same backend.          |
| `utils.inference.FaissKNN`                                  | `utils.retrieval.FaissKNN`                                    | Install `[faiss]` explicitly.                                                                       |
| `AccuracyCalculator.get_accuracy` casting labels to float32 | Labels keep their dtype                                       | Large integer labels no longer collide; embeddings keep their floating dtype.                       |
| `utils.inference.FaissKMeans`                               | `utils.retrieval.FaissKMeans`                                 | Install `[faiss]`; pass this backend explicitly for clustering metrics.                             |
| `utils.inference.CustomKNN`                                 | `utils.retrieval.CustomKNN`                                   | Exact PyTorch search; no Faiss dependency.                                                          |
| `utils.inference.MatchFinder`                               | `utils.retrieval.MatchFinder`                                 | Supply a threshold in the constructor or call.                                                      |
| `MatchFinder.get_matching_pairs(..., return_tuples=True)`   | An `(num_matches, 2)` tensor                                  | Use `.tolist()` only when the caller needs Python pairs.                                            |
| `MatchFinder.is_match(...)`                                 | A boolean tensor on the input device                          | Normalization and distance power now match matrix comparisons.                                      |
| `utils.inference.run_pca`                                   | Application-owned PCA, such as `torch.pca_lowrank`            | Center and project query/reference embeddings with the same fitted basis.                           |
| `utils.inference.add_to_index_and_search`                   | `utils.retrieval.add_to_index_and_search`                     | Prefer the `FaissKNN` call interface for ordinary search.                                           |
| `utils.inference.convert_to_gpu_index`                      | `FaissKNN.to_gpu_index(index)`                                | Configure the backend's `gpus` in its constructor.                                                  |
| `utils.inference.convert_to_cpu_index`                      | `FaissKNN.to_cpu_index(index)`                                | Use the backend instance that owns the index.                                                       |
| `utils.inference.try_gpu`                                   | `FaissKNN.search(index, query, reference, k)`                 | Backend selection belongs to the backend.                                                           |
| `utils.inference.mask_reshape_knn_idx`                      | `utils.retrieval.mask_reshape_knn_idx`                        | The helper moved with the retrieval implementation.                                                 |
| `utils.inference.return_results`                            | `utils.retrieval.return_results`                              | The helper moved with the retrieval implementation.                                                 |
| `utils.inference.get_topk`                                  | `utils.retrieval.get_topk`                                    | The helper moved; ordinary callers use `CustomKNN`.                                                 |
| `AccuracyCalculator()` with implicit Faiss                  | Default PyTorch squared-Euclidean search                      | Pass `knn_func=FaissKNN()` explicitly to retain Faiss; tie order may differ.                        |
| Default `NMI` and `AMI` metrics                             | Explicit `include` plus `kmeans_func`                         | Install `[clustering]` and choose a clustering backend.                                             |
| `AccuracyCalculator(device=None)` selecting CUDA            | The query tensor's device                                     | Move query/reference tensors explicitly or pass `device`.                                           |
| A subclass's automatically discovered `calculate_*` method  | Register the callable in `metric_functions()`                 | [Custom accuracy metrics](accuracy_calculation.md#adding-custom-accuracy-metrics).                  |
| `AccuracyCalculator.description()`                          | Application-owned metric/run labels                           | Read `avg_of_avgs` if the application needs that setting in its label.                              |
| Evaluation with string labels                               | Convert labels to integer tensors in caller code              | Use one mapping shared by query and reference sets.                                                 |

An explicit query/reference evaluation replaces tester-owned split handling:

```python
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

reference = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
reference_labels = torch.tensor([0, 0, 1, 1])
query = torch.tensor([[0.05, 0.0], [3.05, 0.0]])
query_labels = torch.tensor([0, 1])
calculator = AccuracyCalculator(include=("precision_at_1",), k=10)
metrics = calculator.get_accuracy(query, query_labels, reference, reference_labels)
assert metrics["precision_at_1"] == 1.0
```

### Tensor, label, and sampling helpers

| v2 symbol or call                                                          | v3 replacement                                                 | Note                                                                                                |
| -------------------------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `utils.common_functions.to_device`                                         | `tensor.to(device=device, dtype=dtype)`                        | Omit `dtype` to preserve it.                                                                        |
| `utils.common_functions.to_dtype`                                          | `tensor.to(dtype=dtype)`                                       | Explicit conversions also apply inside autocast.                                                    |
| `utils.common_functions.numpy_to_torch`                                    | `torch.as_tensor(array)`                                       | Specify device and dtype when needed.                                                               |
| `utils.common_functions.to_numpy`                                          | `tensor.detach().cpu().numpy()`                                | Omit `detach()` only for tensors without gradients.                                                 |
| `utils.common_functions.is_list_or_tuple`                                  | `isinstance(value, (list, tuple))`                             | Use Python directly.                                                                                |
| `utils.common_functions.return_input`                                      | Return the value directly                                      | Use `torch.nn.Identity` when a module is required.                                                  |
| `utils.common_functions.get_hierarchy_label`                               | Select the desired label column explicitly                     | For a label matrix, use `labels[:, level]`.                                                         |
| `utils.common_functions.map_labels`                                        | Application-owned label mapping                                | Share one mapping across evaluated splits.                                                          |
| `utils.common_functions.process_label`                                     | Select the label column and apply the application mapping      | Hierarchy handling is explicit.                                                                     |
| `utils.common_functions.make_label_to_rank_dict`                           | `{label: i for i, label in enumerate(sorted(unique_labels))}`  | Persist the mapping if later data must use it.                                                      |
| `utils.common_functions.get_label_map`                                     | Construct one mapping per hierarchy level                      | The library no longer owns dataset metadata.                                                        |
| `utils.common_functions.LabelMapper`                                       | Application-owned label mapping                                | Do not build independent query/reference encodings.                                                 |
| `utils.common_functions.NUMPY_RANDOM`                                      | A `torch.Generator` passed to a sampler                        | Seeded sequences differ from version 2.                                                             |
| `utils.common_functions.safe_random_choice`                                | `torch.randperm` or `torch.randint`                            | Select without or with replacement explicitly.                                                      |
| `utils.common_functions.get_labels_to_indices`                             | Group indices with label comparisons                           | For label `c`, use `(labels == c).nonzero(as_tuple=True)[0]`.                                       |
| `utils.common_functions.try_next_on_generator`                             | Own the iterator and handle `StopIteration`                    | Recreate the iterator when the application wants another pass.                                      |
| `utils.common_functions.EmbeddingDataset`                                  | `torch.utils.data.TensorDataset(embeddings, labels)`           | DataLoader returns the paired tensors.                                                              |
| `utils.common_functions.shift_indices_tuple`                               | Offset reference indices explicitly                            | For triplets shift positive/negative indices; for pairs shift their second/fourth tensors.          |
| `utils.common_functions.unslice_by_n`                                      | `torch.stack(parts, dim=1).flatten(0, 1)`                      | Interleave equal-shaped streams along the batch dimension.                                          |
| `utils.common_functions.torch_standard_scaler`                             | Explicit centering and scaling                                 | Fit statistics on the reference/training data and reuse them for queries.                           |
| `utils.common_functions.exclude`                                           | A Python comprehension                                         | Make the filtering rule explicit.                                                                   |
| `utils.common_functions.append_map`                                        | A Python comprehension or `map`                                | Apply the application's transformation directly.                                                    |
| `utils.common_functions.use_cuda_if_available`                             | Select a `torch.device` in caller code                         | Library defaults no longer choose CUDA globally.                                                    |
| `utils.common_functions.TorchInitWrapper`                                  | An in-place `torch.nn.init` callable                           | Bind options with `functools.partial`; see [parameter placement](#loss-parameter-placement).        |
| `utils.common_functions.assert_distance_type`                              | An `isinstance` check on `self.distance` in your module        | The library's distance checks are private; see [custom losses](extend/losses.md#more-on-distances). |
| `utils.loss_and_miner_utils`                                               | `utils.indexing`                                               | Public pair, triplet, and weight utilities; `logsumexp` and other internals are private.            |
| `utils.common_functions`                                                   | Removed from the public API                                    | Helpers used by the library are private; the rows below name replacements.                          |
| `utils.loss_and_miner_utils.get_all_triplets_indices_vectorized_method`    | `utils.indexing.get_all_triplets_indices(labels, ref_labels)`  | Call the label-based API instead of choosing an enumeration implementation.                         |
| `utils.loss_and_miner_utils.get_all_triplets_indices_loop_method`          | `utils.indexing.get_all_triplets_indices(labels, ref_labels)`  | The implementation bounds intermediate memory automatically.                                        |
| `utils.loss_and_miner_utils.get_random_triplet_indices(t_per_anchor=None)` | `utils.indexing.get_random_triplet_indices(t_per_anchor=None)` | Vectorized sampling; `None` draws one triplet per positive of each anchor.                          |
| `utils.common_functions.torch_arange_from_size(x)`                         | `torch.arange(len(x), device=x.device)`                        | Write the index tensor directly.                                                                    |
| `utils.loss_and_miner_utils.repeat_to_match_size`                          | `tensor.repeat(repeats)[:size]` when repetition is intended    | Handle empty inputs explicitly; repeated entries change weighting.                                  |
| `utils.loss_and_miner_utils.matched_size_indices`                          | Choose positive/negative pairing in caller code                | There is no automatic repetition helper.                                                            |

### Logging, model lifecycle, and extension bases

| v2 symbol or call                                                           | v3 replacement                                                                      | Note                                                                      |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `utils.common_functions.LOGGER`                                             | `logging.getLogger("pytorch_metric_learning")`                                      | [Application logging setup](retrieval.md#logging).                        |
| `utils.common_functions.LOGGER_NAME`                                        | The standard `pytorch_metric_learning` logger namespace                             | Individual modules use their own module names.                            |
| `utils.common_functions.set_logger_name`                                    | Configure Python logging handlers and levels                                        | No library logging configuration helper remains.                          |
| `utils.common_functions.COLLECT_STATS`                                      | Application diagnostics                                                             | No global statistics flag remains.                                        |
| `utils.common_functions.add_to_recordable_attributes`                       | Application diagnostics or forward hooks                                            | The record-keeper attribute registry was removed.                         |
| `utils.common_functions.reset_stats`                                        | Reset application-owned accumulators                                                | Modules no longer retain statistics.                                      |
| `utils.common_functions.list_of_recordable_attributes_list_names`           | Application-owned metric names                                                      | The recordable-attribute lists were removed.                              |
| `utils.common_functions.set_requires_grad`                                  | `module.requires_grad_(enabled)`                                                    | Manage optimizer membership and model mode separately.                    |
| `utils.common_functions.set_layers_to_eval`                                 | Call `.eval()` on explicitly selected modules                                       | Select modules by reference or type, not class-name substrings.           |
| `utils.common_functions.get_train_dataloader`                               | `torch.utils.data.DataLoader`                                                       | Pass a sampler or batch sampler according to its PyTorch contract.        |
| `utils.common_functions.get_eval_dataloader`                                | `torch.utils.data.DataLoader`                                                       | Choose evaluation batching and ordering explicitly.                       |
| `utils.common_functions.modelpath_creator`                                  | `pathlib.Path` and application naming rules                                         | The application owns checkpoint paths.                                    |
| `utils.common_functions.save_model`                                         | `torch.save(model.state_dict(), path)`                                              | For a DDP-wrapped model, save the underlying `model.module` state.        |
| `utils.common_functions.load_model`                                         | `model.load_state_dict(torch.load(path, map_location=device, weights_only=True))`   | Handle legacy key conversions explicitly outside the library.             |
| `utils.common_functions.operate_on_dict_of_models`                          | Iterate over the application's model dictionary                                     | No generic checkpoint-operation dispatcher remains.                       |
| `utils.common_functions.save_dict_of_models`                                | Save a dictionary of model `state_dict()` values                                    | Include optimizer and schedule state when resuming training.              |
| `utils.common_functions.load_dict_of_models`                                | Load each state dictionary explicitly                                               | The application owns missing-key and resume policy.                       |
| `utils.common_functions.delete_dict_of_models`                              | `pathlib.Path.unlink` on chosen checkpoint files                                    | The application owns retention rules.                                     |
| `utils.common_functions.latest_version`                                     | Application checkpoint selection                                                    | Select naming, best-model, and resume rules explicitly.                   |
| `utils.common_functions.regex_wrapper`                                      | Python's `re` module                                                                | No library wrapper remains.                                               |
| `utils.common_functions.regex_replace`                                      | `re.sub`                                                                            | Apply replacement rules explicitly.                                       |
| `utils.common_functions.sqlite_obj_to_dict`                                 | Application database handling                                                       | SQLite/record-keeper integration was removed.                             |
| `utils.key_checker.KeyChecker`                                              | Explicit application argument validation                                            | Trainer key schemas were removed.                                         |
| `utils.key_checker.KeyCheckerDict`                                          | Ordinary dictionaries and explicit validation                                       | No trainer schema container remains.                                      |
| `utils.key_checker.default_important`                                       | Application validation rules                                                        | This trainer-schema callback was removed.                                 |
| `utils.logging_presets.HookContainer`                                       | Application callbacks and logging                                                   | The library no longer owns training events.                               |
| `utils.logging_presets.EmptyContainer`                                      | Omit callbacks that are not needed                                                  | No no-op hook container is required.                                      |
| `utils.logging_presets.get_record_keeper`                                   | Configure the application's chosen logger                                           | record-keeper is no longer a dependency.                                  |
| `utils.logging_presets.get_hook_container`                                  | Application callbacks                                                               | See the [training recipes](recipes/training.md).                          |
| `utils.loss_tracker.LossTracker`                                            | Retain returned loss values in caller code                                          | Detach values intended only for logging.                                  |
| `utils.module_with_records.ModuleWithRecords`                               | `torch.nn.Module`                                                                   | Statistics registration was removed.                                      |
| `utils.module_with_records_and_reducer.ModuleWithRecordsAndReducer`         | `torch.nn.Module` with a keyword-only `reducer` option                              | `BaseRegularizer` declares `distance` and `reducer` directly.             |
| `utils.module_with_records_and_reducer.ModuleWithRecordsAndDistance`        | `torch.nn.Module` with a keyword-only `distance` option                             | `BaseMiner` declares `distance` directly.                                 |
| `utils.module_with_records_and_reducer.ModuleWithRecordsReducerAndDistance` | `BaseMetricLossFunction` keyword-only options                                       | See [loss options](losses.md#loss-options); no composition mixins remain. |
| `utils.distributed.all_gather`                                              | PyTorch collectives for raw tensors; library distributed wrappers for losses/miners | [Distributed contracts](distributed.md) specify gradient ownership.       |
| `utils.distributed.all_gather_embeddings_and_labels`                        | `DistributedLossWrapper` or `DistributedMinerWrapper`                               | Use the wrappers to preserve their batch, label, and gradient contracts.  |

## Losses

| v2 symbol or call                                                                        | v3 replacement                                                           | Note                                                                                            |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `losses.BaseLossWrapper`                                                                 | `torch.nn.Module`                                                        | `CrossBatchMemory` and `SelfSupervisedLoss` inherit it directly.                                |
| `losses.EmbeddingRegularizerMixin`                                                       | `embedding_regularizer` and `embedding_reg_weight` on the loss           | The same removal applies to `losses.mixins.EmbeddingRegularizerMixin`.                          |
| `losses.WeightRegularizerMixin`                                                          | Keyword-only `weight_regularizer` and `weight_reg_weight`                | The same removal applies to `losses.mixins.WeightRegularizerMixin`.                             |
| `losses.mixins.WeightMixin`                                                              | A loss parameter plus `regularization_terms()` when needed               | Weight initialization accepts an in-place callable.                                             |
| `losses.proxy_losses.ProxyNCALoss`                                                       | `losses.proxy_nca_loss.ProxyNCALoss`                                     | The normal `losses.ProxyNCALoss` import is unchanged.                                           |
| `losses.dynamic_soft_margin_loss.find_hard_negatives`                                    | `DynamicSoftMarginLoss` for its built-in mining, or an explicit miner    | The loss no longer exposes a separate mining helper.                                            |
| Automatic parameter movement/casting during `forward`                                    | `loss_func.to(device, dtype)` before use                                 | Include loss parameters in the optimizer; see the example below.                                |
| `BaseMetricLossFunction._sub_loss_names()`                                               | The `sub_loss_names` class attribute                                     | A tuple of objective term names; penalties are added separately.                                |
| `BaseMetricLossFunction.get_default_distance()` / `get_default_reducer()`                | `default_distance()` / `default_reducer()`                               | The same applies to miners and regularizers.                                                    |
| `BaseMetricLossFunction.zero_losses()`                                                   | `zero_losses(embeddings)`                                                | Zero terms are differentiable tensors connected to the embeddings.                              |
| `SomeLoss(..., **kwargs)` forwarding `distance`, `reducer`, and regularizer arguments    | `SomeLoss(..., **options)` typed by `types.LossOptions`                  | The same four keyword-only options; see [loss options](losses.md#loss-options).                 |
| `reducer=` deep-copied and wrapped in `MultipleReducers` for multi-term losses           | The reducer object is used as given for every term                       | Pass a `MultipleReducers` to vary reducers per term.                                            |
| `reducers.DoNothingReducer()` to get the loss dictionary                                 | `loss_func.loss_terms(embeddings, labels, ...)`                          | `forward` always returns a tensor; see [unreduced terms](losses.md#unreduced-loss-terms).       |
| `BaseMetricLossFunction.all_regularization_loss_names()`                                 | `regularization_terms(embeddings)`                                       | Regularization is composed explicitly after objective reduction.                                |
| `GenericPairLoss(mat_based_loss=...)` with `_compute_loss(...)`                          | `GenericPairLoss(**options)` with `pair_loss(mat, pairs)`                | Use `pair_values` or `pair_masks`; see [pair-based losses](extend/losses.md#pair-based-losses). |
| `CrossBatchMemory.check_loss_support` / `supported_losses()`                             | Any `BaseMetricLossFunction` that accepts `ref_emb`                      | Losses that reject references raise `ValueError` on the first call.                             |
| `SelfSupervisedLoss.check_loss_support` / `supported_losses()`                           | Any `BaseMetricLossFunction`; asymmetric mode needs `ref_emb` support    | Other modules raise `TypeError` at construction.                                                |
| `loss.W`, `loss.proxies`, `loss.fc` parameters                                           | `loss.weight` with shape `(num_classes, embedding_size)`                 | One row per class or center; see [learnable weights](losses.md#learnable-weight-arguments).     |
| `ArcFaceLoss(margin=28.6)` and `SubCenterArcFaceLoss(margin=28.6)` in degrees            | `margin=0.5` in radians                                                  | The paper's unit; convert with `math.radians`.                                                  |
| `LargeMarginSoftmaxLoss(margin=4.0)` float margins                                       | An `int` margin                                                          | Non-integer margins raise `ValueError` instead of being truncated.                              |
| `P2SGradLoss(descriptors_dim, num_classes)`                                              | `P2SGradLoss(num_classes, embedding_size)`                               | Argument order matches the other classification losses.                                         |
| `VICRegLoss(...)(embeddings, ref_emb=ref_emb)` with `reducer` or `distance`              | `VICRegLoss(...)(embeddings, ref_emb)`                                   | A plain module without loss options; the value is unchanged.                                    |
| `MultipleLosses([loss_a, loss_b], weights=[1, 0.5])`                                     | `MultipleLosses({"a": loss_a, "b": loss_b}, weights={"a": 1, "b": 0.5})` | Mappings only; see [MultipleLosses](losses/multiple_losses.md).                                 |
| `CrossBatchMemory` moving its queue during `forward`                                     | Place the wrapper with `.to(device, dtype)`                              | `label_dtype` is keyword-only and defaults to `torch.long`.                                     |
| `CrossBatchMemory` checkpoint `_extra_state`                                             | A v3 checkpoint with `queue_state`                                       | Old queue metadata is not loaded through a compatibility path.                                  |
| Fixed `MarginLoss.beta` outside its state dictionary                                     | A checkpointed buffer that follows `.to()`                               | Account for the added `beta` key when importing an old checkpoint.                              |
| `DynamicSoftMarginLoss` moving its histogram during `forward`                            | Place the loss with `.to(device, dtype)`                                 | Its checkpointed histogram keeps its configured precision.                                      |
| `SubCenterArcFaceLoss.get_outliers(..., return_dominant_centers=False)` returning a pair | The outlier tensor alone                                                 | Request dominant centers explicitly when needed.                                                |
| `ManifoldLoss(l=..., K=..., lambdaC=...)`                                                | `ManifoldLoss(embedding_size=..., num_proxies=..., lambda_c=...)`        | Parameter meanings are unchanged.                                                               |
| `MultipleLosses.get_indices_tuple`                                                       | Pass a shared `indices_tuple` to `forward`, or configure per-loss miners | These choices are mutually exclusive; [MultipleLosses contract](losses/multiple_losses.md).     |
| `MultipleLosses.assertions_if_not_none`                                                  | Constructor validation                                                   | Invalid keys raise explicit exceptions.                                                         |
| `ArcFaceLoss.cast_types`                                                                 | `.to(device, dtype)`                                                     | Placement belongs to the caller.                                                                |
| `CosFaceLoss.cast_types`                                                                 | `.to(device, dtype)`                                                     | Placement belongs to the caller.                                                                |
| `LargeMarginSoftmaxLoss.cast_types`                                                      | `.to(device, dtype)`                                                     | This also applies to inherited classification losses.                                           |
| `NormalizedSoftmaxLoss.cast_types`                                                       | `.to(device, dtype)`                                                     | Placement belongs to the caller.                                                                |
| `ProxyAnchorLoss.cast_types`                                                             | `.to(device, dtype)`                                                     | Placement belongs to the caller.                                                                |
| `SoftTripleLoss.cast_types`                                                              | `.to(device, dtype)`                                                     | Placement belongs to the caller.                                                                |
| `ProxyAnchorLoss.get_default_weight_init_func`                                           | The `weight_init_func` constructor argument                              | Omit it to retain the loss's default initializer.                                               |
| `SoftTripleLoss.get_default_weight_init_func`                                            | The `weight_init_func` constructor argument                              | Omit it to retain the loss's default initializer.                                               |
| `AngularLoss.set_stats`                                                                  | Compute diagnostics in caller code                                       | No recorded `avg_angle` state remains.                                                          |
| `TupletMarginLoss.set_stats`                                                             | Compute diagnostics in caller code                                       | The loss returns the objective without statistics side effects.                                 |

### Loss parameter placement

```python
from functools import partial
from itertools import chain

import torch
from pytorch_metric_learning import losses, regularizers

torch.manual_seed(7)
model = torch.nn.Linear(4, 3).to(dtype=torch.float64)
loss_func = losses.ProxyNCALoss(
    num_classes=2,
    embedding_size=3,
    weight_regularizer=regularizers.LpRegularizer(),
    weight_reg_weight=0.01,
    weight_init_func=partial(torch.nn.init.normal_, std=0.1),
).to(dtype=torch.float64)
optimizer = torch.optim.SGD(chain(model.parameters(), loss_func.parameters()), lr=0.01)
inputs = torch.randn(4, 4, dtype=torch.float64)
labels = torch.tensor([0, 0, 1, 1])
optimizer.zero_grad(set_to_none=True)
loss = loss_func(model(inputs), labels)
loss.backward()
assert torch.isfinite(loss)
assert torch.isfinite(model.weight.grad).all()
assert torch.isfinite(next(loss_func.parameters()).grad).all()
optimizer.step()
```

## Miners

| v2 symbol or call                                    | v3 replacement                                            | Note                                                                        |
| ---------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| `miners.EmbeddingsAlreadyPackagedAsTriplets`         | Construct the three index tensors directly                | See the packed-triplet example below.                                       |
| `BaseMiner.output_assertion`                         | Automatic validation in `BaseMiner.forward`               | Custom miners only implement `mine`; see [custom miners](extend/miners.md). |
| `SomeMiner(..., **kwargs)` forwarding `distance`     | `SomeMiner(..., *, distance=None)`                        | `distance` is the only miner option.                                        |
| `BatchEasyHardMiner.HARD`, `SEMIHARD`, `EASY`, `ALL` | The strings `"hard"`, `"semihard"`, `"easy"`, `"all"`     | The class constants were removed.                                           |
| `HDCMiner.set_idx_externally`                        | `miner(..., candidates=pairs)`                            | Candidates belong to one call.                                              |
| `HDCMiner.reset_idx`                                 | Omit `candidates` on the next call                        | No candidate state persists.                                                |
| `HDCMiner.set_idx`                                   | Pass labels and optional candidates to the miner          | The caller no longer initializes miner state.                               |
| `HDCMiner.filter_original_indices`                   | Use the tuple returned by the miner                       | Filtering no longer mutates stored candidates.                              |
| `HDCMiner.should_select_largest`                     | The miner handles orientation from `distance.is_inverted` | Custom selection should make its similarity/distance ordering explicit.     |
| `AngularMiner.set_stats`                             | Diagnostics computed from returned indices                | No stored miner statistics remain.                                          |
| `BatchEasyHardMiner.set_stats`                       | Diagnostics computed from returned indices                | No stored miner statistics remain.                                          |
| `BatchEasyHardMiner.get_func_for_stats`              | Explicit application reductions                           | Choose `min`, `max`, or `mean` for the desired diagnostic.                  |
| `PairMarginMiner.set_stats`                          | Diagnostics computed from returned indices                | No stored miner statistics remain.                                          |
| `TripletMarginMiner.set_stats`                       | Diagnostics computed from returned indices                | No stored miner statistics remain.                                          |

### Consecutively packed triplets

```python
import torch
from pytorch_metric_learning import losses

embeddings = torch.tensor(
    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
     [0.0, 1.0], [0.1, 0.9], [1.0, 0.0]],
    requires_grad=True,
)
labels = torch.tensor([0, 0, 1, 1, 1, 0])
indices_tuple = torch.arange(len(embeddings), device=embeddings.device).reshape(-1, 3).unbind(1)
loss = losses.TripletMarginLoss(margin=1.4)(embeddings, labels, indices_tuple)
loss.backward()
assert loss > 0
assert torch.isfinite(embeddings.grad).all()
```

### Per-call cascade candidates

```python
import torch
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import indexing

embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
labels = torch.tensor([0, 0, 1, 1])
candidates = indexing.get_all_pairs_indices(labels)
miner = miners.HDCMiner(filter_percentage=0.5)
pairs = miner(embeddings, labels, candidates=candidates)
assert len(pairs[0]) == 2
assert len(pairs[2]) == 4
assert torch.equal(labels[pairs[0]], labels[pairs[1]])
assert torch.all(labels[pairs[2]] != labels[pairs[3]])
```

## Samplers

| v2 symbol or call                  | v3 replacement                                                     | Note                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `samplers.TuplesToWeightsSampler`  | `convert_to_weights` plus `torch.utils.data.WeightedRandomSampler` | Compute full-dataset weights in caller code; [weighted-sampling recipe](recipes/training.md#mining-weighted-sampling-contract). |
| NumPy-seeded `MPerClassSampler`    | Pass a keyword-only `torch.Generator`                              | `torch.manual_seed` also controls its default generator.                                                                        |
| NumPy-seeded `HierarchicalSampler` | Pass a keyword-only `torch.Generator`                              | Seeded sequences differ from version 2.                                                                                         |
| NumPy-seeded `FixedSetOfTriplets`  | Pass a keyword-only `torch.Generator`                              | Seeded sequences differ from version 2.                                                                                         |
| `HierarchicalSampler.reshuffle()`  | Begin a new iteration over the sampler                             | Batch generation occurs during iteration, not construction or `len()`.                                                          |

## Distances

| v2 symbol or call                                                    | v3 replacement                                               | Note                                                                                                |
| -------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `BaseDistance(is_inverted=True)`                                     | Set `is_inverted = True` on the custom similarity class      | The constructor no longer accepts this flag; `p` still controls normalization.                      |
| `BaseDistance.largest_dist`                                          | Explicit `torch.minimum`/`torch.maximum` using `is_inverted` | For similarities, the largest distance uses the smaller similarity; distances use the larger value. |
| `BaseDistance.set_default_stats`                                     | Application diagnostics                                      | No recorded norm statistics remain.                                                                 |
| `BatchedDistance(..., iter_fn=callback)`                             | Iterate over `distance(query, ref)`                          | Each item is `(mat, start, end)`; no callback is involved.                                          |
| `CosineSimilarity(normalize_embeddings=...)`                         | `CosineSimilarity(p=2, power=1)`                             | Cosine similarity always normalizes.                                                                |
| Forwarded attributes such as `batched_distance.normalize_embeddings` | `batched_distance.distance.normalize_embeddings`             | Only `is_inverted` is exposed directly for search ordering.                                         |

The iterator yields the matrix and query slice bounds for each block:

```python
import torch
from pytorch_metric_learning import distances

query = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
reference = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
output = query.new_empty(len(query), len(reference))

distance = distances.BatchedDistance(
    distances.LpDistance(normalize_embeddings=False), batch_size=2
)
for matrix, start, end in distance(query, reference):
    output[start:end] = matrix
expected = query.new_tensor([[1.0, 2.0], [2**0.5, 1.0], [1.0, 8**0.5]])
torch.testing.assert_close(output, expected)
```

## Reducers

| v2 symbol or call                                                          | v3 replacement                                           | Note                                                                                     |
| -------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `BaseReducer.assert_sizes*`                                                | `reducers.base_reducer.validate_term(term)`              | One validator raises `TypeError` or `ValueError` for every reduction type.               |
| `BaseReducer.get_reduction_func`                                           | Override the named reduction methods                     | `reduce(term, embeddings, labels)` dispatches on the reduction type.                     |
| `element_reduction(losses, indices, embeddings, labels, **kwargs)`         | `element_reduction(losses, indices, embeddings, labels)` | Divisors are read by `DivisorReducer.reduce`; hooks take no extra arguments.             |
| `BaseReducer.forward` requiring exactly one term                           | `forward` reduces every term and sums them               | `MultipleReducers` chooses a reducer per term name.                                      |
| `reducers.DoNothingReducer`                                                | `loss_func.loss_terms(...)`                              | Reducers always return a tensor.                                                         |
| `{"losses": 0}` zero terms                                                 | A zero tensor, or an empty `losses` tensor               | `losses` is always a tensor; empty tensors reduce to a differentiable zero.              |
| `ThresholdReducer` boolean indexing                                        | Masked sum and count                                     | Values are unchanged; the reduction no longer synchronizes with the host.                |
| `BaseReducer.set_losses_size_stat`                                         | Compute counts from the loss dictionary                  | No recorded reducer statistics remain.                                                   |
| `ClassWeightedReducer.element_reduction_helper`                            | `element_reduction`                                      | Default tuple reductions pass anchor indices to this method.                             |
| `ThresholdReducer.element_reduction_helper`                                | `element_reduction`                                      | Default tuple reductions pass anchor indices to this method.                             |
| `PerAnchorReducer.tuple_reduction_helper`                                  | The reducer's public `forward`/reduction methods         | Custom tuple behavior belongs in explicit pair/triplet overrides.                        |
| `ThresholdReducer.set_stats`                                               | Compute diagnostics from caller-owned loss terms         | No stored `num_past_filter` remains.                                                     |
| `MultipleReducers` entries named `embedding_reg_loss` or `weight_reg_loss` | Configure regularizers on the loss                       | Penalties are added after objective reduction; these entries are no longer used.         |
| Custom `already_reduced` handling of regularization penalties              | Customize the loss's `regularization_terms()`            | A reducer never receives those penalties; `loss_terms()` returns them in the dictionary. |

## Regularizers

| v2 symbol or call                                                    | v3 replacement                                                        | Note                                                                     |
| -------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `CenterInvariantRegularizer.compute_loss(weights=...)`               | `compute_loss(x=...)`                                                 | Prefer calling the regularizer on the tensor directly.                   |
| `LpRegularizer.compute_loss(embeddings=...)`                         | `compute_loss(x=...)`                                                 | Positional calls are unchanged.                                          |
| `RegularFaceRegularizer.compute_loss(weights=...)`                   | `compute_loss(x=...)`                                                 | Positional calls are unchanged.                                          |
| `SparseCentersRegularizer.compute_loss(weights=...)`                 | `compute_loss(x=...)`                                                 | Positional calls are unchanged.                                          |
| `ZeroMeanRegularizer.compute_loss(embeddings=...)`                   | `compute_loss(x=...)`                                                 | Positional calls are unchanged.                                          |
| `SparseCentersRegularizer.set_stats`                                 | Application diagnostics                                               | Its same-class mask is now a non-persistent buffer that follows `.to()`. |
| `SomeRegularizer(..., **kwargs)` forwarding `distance` and `reducer` | `SomeRegularizer(..., **options)` typed by `types.RegularizerOptions` | Both options are keyword-only.                                           |

The loss dictionary and pair/triplet tuple formats remain runtime dictionaries
and tuples. Their public type definitions are in `pytorch_metric_learning.types`;
no conversion wrapper is required.
