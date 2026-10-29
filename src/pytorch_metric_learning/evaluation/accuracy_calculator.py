import importlib
import logging
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..distances import LpDistance
from ..types import (
    ArrayLike,
    KMeansFunction,
    KNNFunction,
    LabelComparison,
    MetricNeighbors,
    NeighborLimit,
)
from .retrieval import CustomKNN

__all__ = [
    "AccuracyCalculator",
    "ClusteringMetric",
    "RetrievalInputs",
    "RetrievalMetric",
    "mean_average_precision",
    "mean_average_precision_at_r",
    "mean_reciprocal_rank",
    "precision_at_1",
    "precision_at_k",
    "r_precision",
]
_LOGGER = logging.getLogger(__name__)
_DEFAULT_KNN_BATCH_SIZE = 256
_KMEANS_REQUIRED = (
    "Clustering metrics such as NMI and AMI require kmeans_func, for example "
    "FaissKMeans() from the faiss extra or a function returning one cluster "
    "index per row"
)


@dataclass(frozen=True)
class RetrievalInputs:
    query_labels: Tensor
    knn_labels: Tensor
    matches: Tensor
    relevant_counts: Tensor


@dataclass(frozen=True)
class RetrievalMetric:
    function: Callable[[RetrievalInputs], Tensor]
    neighbors: MetricNeighbors = "all"

    def __post_init__(self) -> None:
        valid_count = (
            isinstance(self.neighbors, int)
            and not isinstance(self.neighbors, bool)
            and self.neighbors > 0
        )
        if not (valid_count or self.neighbors in ("relevant", "all")):
            raise ValueError(
                "neighbors must be a positive integer, 'relevant', or 'all'"
            )


@dataclass(frozen=True)
class ClusteringMetric:
    function: Callable[[Tensor, Tensor], float]


type _Metric = RetrievalMetric | ClusteringMetric


def _mutual_information_scores() -> Any:
    try:
        return importlib.import_module("sklearn.metrics")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "NMI and AMI require scikit-learn. Install the optional dependency with "
            "`pip install pytorch-metric-learning[clustering]`."
        ) from error


def _equal_label_counts(labels: Tensor, reference_labels: Tensor) -> Tensor:
    dtype = torch.promote_types(labels.dtype, reference_labels.dtype)
    labels, reference_labels = labels.to(dtype), reference_labels.to(dtype)
    values, counts = torch.unique(reference_labels, return_counts=True)
    if len(values) == 0:
        return torch.zeros(len(labels), dtype=torch.long, device=labels.device)
    positions = torch.searchsorted(values, labels).clamp_max(len(values) - 1)
    return torch.where(values[positions] == labels, counts[positions], 0)


def _relevant_counts(
    unique_labels: Tensor,
    reference_labels: Tensor,
    ref_includes_query: bool,
    label_comparison_fn: LabelComparison | None,
) -> Tensor:
    if label_comparison_fn is None:
        counts = _equal_label_counts(unique_labels, reference_labels)
    else:
        counts = torch.stack(
            [
                label_comparison_fn(unique_labels[i : i + 1], reference_labels).sum()
                for i in range(len(unique_labels))
            ]
        ).to(torch.long)
    if ref_includes_query:
        comparison = label_comparison_fn or torch.eq
        counts = counts - comparison(unique_labels, unique_labels).to(torch.long)
    return counts


def _relevance_mask(counts_per_query: Tensor, num_k: int) -> Tensor:
    positions = torch.arange(num_k, device=counts_per_query.device)
    return positions < counts_per_query.unsqueeze(1)


def precision_at_k(inputs: RetrievalInputs, k: int) -> Tensor:
    return inputs.matches[:, :k].sum(dim=1, dtype=torch.float64) / k


def precision_at_1(inputs: RetrievalInputs) -> Tensor:
    return precision_at_k(inputs, 1)


def r_precision(inputs: RetrievalInputs) -> Tensor:
    mask = _relevance_mask(inputs.relevant_counts, inputs.matches.shape[1])
    hits = (inputs.matches & mask).sum(dim=1, dtype=torch.float64)
    return hits / inputs.relevant_counts


def _average_precision(matches: Tensor, counts: Tensor) -> Tensor:
    ranks = torch.arange(
        1, matches.shape[1] + 1, device=matches.device, dtype=torch.float64
    )
    precision = torch.cumsum(matches, dim=1, dtype=torch.float64) / ranks
    return (precision * matches).sum(dim=1) / counts


def mean_average_precision(inputs: RetrievalInputs) -> Tensor:
    return _average_precision(inputs.matches, inputs.relevant_counts)


def mean_average_precision_at_r(inputs: RetrievalInputs) -> Tensor:
    mask = _relevance_mask(inputs.relevant_counts, inputs.matches.shape[1])
    return _average_precision(inputs.matches & mask, inputs.relevant_counts)


def mean_reciprocal_rank(inputs: RetrievalInputs) -> Tensor:
    matches = inputs.matches
    if matches.shape[1] == 0:
        return torch.zeros(len(matches), dtype=torch.float64, device=matches.device)
    first_match = torch.argmax(matches.to(torch.uint8), dim=1).to(torch.float64) + 1
    return torch.where(matches.any(dim=1), 1 / first_match, 0.0)


def _normalized_mutual_information(
    query_labels: Tensor, cluster_labels: Tensor
) -> float:
    return float(
        _mutual_information_scores().normalized_mutual_info_score(
            *_host_arrays(query_labels, cluster_labels)
        )
    )


def _adjusted_mutual_information(query_labels: Tensor, cluster_labels: Tensor) -> float:
    return float(
        _mutual_information_scores().adjusted_mutual_info_score(
            *_host_arrays(query_labels, cluster_labels)
        )
    )


_BUILTIN_METRICS: Mapping[str, _Metric] = MappingProxyType(
    {
        "AMI": ClusteringMetric(_adjusted_mutual_information),
        "NMI": ClusteringMetric(_normalized_mutual_information),
        "mean_average_precision": RetrievalMetric(mean_average_precision),
        "mean_average_precision_at_r": RetrievalMetric(
            mean_average_precision_at_r, "relevant"
        ),
        "mean_reciprocal_rank": RetrievalMetric(mean_reciprocal_rank),
        "precision_at_1": RetrievalMetric(precision_at_1, 1),
        "r_precision": RetrievalMetric(r_precision, "relevant"),
    }
)


def _class_key(label: Tensor) -> Hashable:
    return label.item() if label.ndim == 0 else tuple(label.tolist())


@dataclass(frozen=True)
class _QueryGroups:
    inputs: RetrievalInputs
    has_relevant: Tensor
    classes: Tensor
    class_index: Tensor

    def values(self, name: str, metric: RetrievalMetric) -> tuple[Tensor, Tensor]:
        values = metric.function(self.inputs)
        if not isinstance(values, Tensor) or values.shape != self.has_relevant.shape:
            raise ValueError(
                f"Retrieval metric {name!r} must return a tensor of shape "
                f"{tuple(self.has_relevant.shape)}, one value per query"
            )
        values = values.to(torch.float64)
        return values, self.has_relevant & ~torch.isnan(values)

    def class_means(self, values: Tensor, defined: Tensor) -> Tensor:
        num_classes = len(self.classes)
        sums = values.new_zeros(num_classes).index_add(
            0, self.class_index, torch.where(defined, values, 0.0)
        )
        counts = values.new_zeros(num_classes).index_add(
            0, self.class_index, defined.to(torch.float64)
        )
        return sums / counts


def _check_neighbor_indices(indices: Tensor, num_reference: int) -> None:
    if indices.numel() == 0:
        return
    low, high = torch.stack(torch.aminmax(indices)).tolist()
    if low < 0 or high >= num_reference:
        raise ValueError(
            f"knn_func returned neighbor indices from {low} to {high}, but the "
            f"reference set has {num_reference} rows. Indices must address the "
            "reference passed to knn_func; a FaissKNN with reset_before=False "
            "appends that reference to its index on every call."
        )


class AccuracyCalculator:
    def __init__(
        self,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        avg_of_avgs: bool = False,
        k: NeighborLimit = None,
        label_comparison_fn: LabelComparison | None = None,
        device: torch.device | str | None = None,
        knn_func: KNNFunction | None = None,
        kmeans_func: KMeansFunction | None = None,
        custom_metrics: Mapping[str, RetrievalMetric | ClusteringMetric] | None = None,
    ) -> None:
        custom = dict(custom_metrics or {})
        replaced = sorted(custom.keys() & _BUILTIN_METRICS.keys())
        if replaced:
            raise ValueError(
                f"custom_metrics cannot replace built-in metrics: {replaced}"
            )
        for name, metric in custom.items():
            if not isinstance(metric, (RetrievalMetric, ClusteringMetric)):
                raise TypeError(
                    f"custom metric {name!r} must be a RetrievalMetric or "
                    "ClusteringMetric"
                )
        self.available_metrics: Mapping[str, _Metric] = MappingProxyType(
            {**_BUILTIN_METRICS, **custom}
        )
        self.check_metric_names(include, exclude)
        if len(include) == 0:
            include = [
                name
                for name, metric in self.available_metrics.items()
                if isinstance(metric, RetrievalMetric)
            ]
        self.metrics = tuple(x for x in include if x not in exclude)
        self.avg_of_avgs = avg_of_avgs
        self.device = None if device is None else torch.device(device)
        self.knn_func: KNNFunction = (
            CustomKNN(
                LpDistance(normalize_embeddings=False, power=2),
                batch_size=_DEFAULT_KNN_BATCH_SIZE,
            )
            if knn_func is None
            else knn_func
        )
        self.kmeans_func = kmeans_func
        if (not (isinstance(k, int) and k > 0)) and (k not in [None, "max_bin_count"]):
            raise ValueError(
                "k must be an integer greater than 0, or None, or 'max_bin_count'"
            )
        self.k = k
        self.custom_comparison = label_comparison_fn
        self.label_comparison_fn: LabelComparison = label_comparison_fn or torch.eq
        uses_clustering = bool(self.clustering_names(self.metrics))
        if label_comparison_fn is not None and uses_clustering:
            raise ValueError("Clustering metrics require the default label comparison")
        if self.kmeans_func is None and uses_clustering:
            raise ValueError(_KMEANS_REQUIRED)

    def clustering_names(self, names: Sequence[str]) -> list[str]:
        return [
            name
            for name in names
            if isinstance(self.available_metrics[name], ClusteringMetric)
        ]

    def retrieval_metrics(self, names: Sequence[str]) -> dict[str, RetrievalMetric]:
        return {
            name: metric
            for name in names
            if isinstance(metric := self.available_metrics[name], RetrievalMetric)
        }

    def get_cluster_labels(self, query: Tensor, query_labels: Tensor) -> Tensor:
        if self.kmeans_func is None:
            raise ValueError(_KMEANS_REQUIRED)
        num_clusters = len(torch.unique(query_labels, dim=0))
        return self.kmeans_func(query, num_clusters)

    def get_accuracy(
        self,
        query: ArrayLike,
        query_labels: ArrayLike,
        reference: ArrayLike | None = None,
        reference_labels: ArrayLike | None = None,
        ref_includes_query: bool = False,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> dict[str, float]:
        names = self.selected_metrics(include, exclude)
        with torch.no_grad():
            tensors = self.prepare(
                query, query_labels, reference, reference_labels, ref_includes_query
            )
            retrieval = self.retrieval_metrics(names)
            groups = self.retrieve(*tensors, retrieval) if retrieval else None
            cluster_labels = (
                self.get_cluster_labels(tensors[0], tensors[1])
                if self.clustering_names(names)
                else None
            )
            results: dict[str, float] = {}
            for name in names:
                metric = self.available_metrics[name]
                if isinstance(metric, RetrievalMetric) and groups is not None:
                    results[name] = self.reduce(groups, *groups.values(name, metric))
                elif (
                    isinstance(metric, ClusteringMetric) and cluster_labels is not None
                ):
                    results[name] = float(metric.function(tensors[1], cluster_labels))
            return results

    def get_accuracy_per_class(
        self,
        query: ArrayLike,
        query_labels: ArrayLike,
        reference: ArrayLike | None = None,
        reference_labels: ArrayLike | None = None,
        ref_includes_query: bool = False,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> dict[str, dict[Hashable, float]]:
        names = self.selected_metrics(include, exclude)
        clustering = self.clustering_names(include)
        if clustering:
            raise ValueError(
                f"Clustering metrics have no per-class values: {clustering}"
            )
        retrieval = self.retrieval_metrics(names)
        with torch.no_grad():
            tensors = self.prepare(
                query, query_labels, reference, reference_labels, ref_includes_query
            )
            if not retrieval:
                return {}
            groups = self.retrieve(*tensors, retrieval)
            classes = [_class_key(label) for label in groups.classes.cpu()]
            return {
                name: dict(
                    zip(
                        classes,
                        groups.class_means(*groups.values(name, metric)).tolist(),
                        strict=True,
                    )
                )
                for name, metric in retrieval.items()
            }

    def reduce(self, groups: _QueryGroups, values: Tensor, defined: Tensor) -> float:
        if self.avg_of_avgs:
            return torch.nanmean(groups.class_means(values, defined)).item()
        total = torch.where(defined, values, 0.0).sum()
        return (total / defined.sum()).item()

    def selected_metrics(
        self, include: Sequence[str], exclude: Sequence[str]
    ) -> list[str]:
        self.check_metric_names(include, exclude)
        if not set(include) <= set(self.metrics):
            raise ValueError(
                f"include must name metrics selected at construction: {self.metrics}"
            )
        return [name for name in (include or self.metrics) if name not in exclude]

    def prepare(
        self,
        query: ArrayLike,
        query_labels: ArrayLike,
        reference: ArrayLike | None,
        reference_labels: ArrayLike | None,
        ref_includes_query: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        if reference is None:
            reference, reference_labels, ref_includes_query = query, query_labels, True
        if reference_labels is None:
            raise ValueError("reference_labels are required with reference")
        device = self.device or torch.as_tensor(query).device
        query_t, reference_t = _embedding_pair(query, reference, device)
        query_labels_t, reference_labels_t = (
            torch.as_tensor(x).to(device) for x in (query_labels, reference_labels)
        )
        self.check_inputs(
            query_t, query_labels_t, reference_t, reference_labels_t, ref_includes_query
        )
        return (
            query_t,
            query_labels_t,
            reference_t,
            reference_labels_t,
            ref_includes_query,
        )

    def check_inputs(
        self,
        query: Tensor,
        query_labels: Tensor,
        reference: Tensor,
        reference_labels: Tensor,
        ref_includes_query: bool,
    ) -> None:
        if len(query) != len(query_labels) or len(reference) != len(reference_labels):
            raise ValueError("embeddings and labels must have the same length")
        if self.custom_comparison is None and (
            query_labels.ndim != 1 or reference_labels.ndim != 1
        ):
            raise ValueError(
                "Multi-column labels require a label_comparison_fn; the default "
                "comparison supports one label per row"
            )
        if not (torch.isfinite(query).all() and torch.isfinite(reference).all()):
            raise ValueError("query and reference embeddings must be finite")
        if ref_includes_query and not (
            torch.allclose(query, reference[: len(query)])
            and _labels_equal(query_labels, reference_labels[: len(query_labels)])
        ):
            raise ValueError(
                "When ref_includes_query is True, the first len(query) rows of "
                "reference and reference_labels must equal query and query_labels."
            )

    def retrieve(
        self,
        query: Tensor,
        query_labels: Tensor,
        reference: Tensor,
        reference_labels: Tensor,
        ref_includes_query: bool,
        metrics: Mapping[str, RetrievalMetric],
    ) -> _QueryGroups:
        classes, class_index = torch.unique(query_labels, dim=0, return_inverse=True)
        counts = _relevant_counts(
            classes, reference_labels, ref_includes_query, self.custom_comparison
        )
        counts_per_query = counts[class_index]
        has_relevant = counts_per_query > 0
        if not torch.any(has_relevant):
            _LOGGER.warning("None of the query labels are in the reference set.")
        num_k = self.determine_k(counts, len(reference), ref_includes_query, metrics)
        _, knn_indices = self.knn_func(query, num_k, reference, ref_includes_query)
        _check_neighbor_indices(knn_indices, len(reference))
        knn_labels = reference_labels[knn_indices]
        matches = self.label_comparison_fn(query_labels.unsqueeze(1), knn_labels)
        inputs = RetrievalInputs(
            query_labels=query_labels,
            knn_labels=knn_labels,
            matches=matches.to(torch.bool),
            relevant_counts=counts_per_query,
        )
        return _QueryGroups(inputs, has_relevant, classes, class_index)

    def check_metric_names(
        self, include: Sequence[str], exclude: Sequence[str]
    ) -> None:
        names = list(self.available_metrics)
        for metrics in (include, exclude):
            if isinstance(metrics, str):
                raise TypeError("include and exclude must be sequences of metric names")
            if not set(metrics).issubset(names):
                raise ValueError(f"Metrics must be one or more of: {names}.")

    def determine_k(
        self,
        counts: Tensor,
        num_reference_embeddings: int,
        ref_includes_query: bool,
        metrics: Mapping[str, RetrievalMetric],
    ) -> int:
        available = num_reference_embeddings - int(ref_includes_query)
        max_relevant = int(counts.max().item()) if len(counts) else 0
        if self.k == "max_bin_count":
            limit = max(max_relevant, 1)
        elif self.k is None:
            limit = available
        else:
            limit = self.k
            at_r = sorted(n for n, m in metrics.items() if m.neighbors == "relevant")
            if self.k < max_relevant and at_r:
                _LOGGER.warning(
                    "You are computing %s, but k (%s) is less than the largest "
                    "number of relevant references (%s), so these metrics will be "
                    "incorrect. Set k='max_bin_count'. For MAP@%s instead of MAP@R, "
                    "use 'mean_average_precision' rather than "
                    "'mean_average_precision_at_r'.",
                    at_r,
                    self.k,
                    max_relevant,
                    self.k,
                )
        needed = max(
            _neighbors_needed(metric, max_relevant, available)
            for metric in metrics.values()
        )
        return min(limit, needed, available)


def _neighbors_needed(
    metric: RetrievalMetric, max_relevant: int, available: int
) -> int:
    if isinstance(metric.neighbors, int):
        return metric.neighbors
    if metric.neighbors == "relevant":
        return max(max_relevant, 1)
    return available


def _embedding_pair(
    query: ArrayLike, reference: ArrayLike, device: torch.device
) -> tuple[Tensor, Tensor]:
    query_t, reference_t = (torch.as_tensor(x).to(device) for x in (query, reference))
    if not (query_t.is_floating_point() and reference_t.is_floating_point()):
        return query_t.float(), reference_t.float()
    dtype = torch.promote_types(query_t.dtype, reference_t.dtype)
    return query_t.to(dtype), reference_t.to(dtype)


def _host_arrays(*values: ArrayLike) -> list[np.ndarray]:
    return [torch.as_tensor(value).cpu().numpy() for value in values]


def _labels_equal(x: Tensor, y: Tensor) -> bool:
    if x.is_floating_point() or y.is_floating_point():
        dtype = torch.promote_types(x.dtype, y.dtype)
        return torch.allclose(x.to(dtype), y.to(dtype))
    return torch.equal(x, y)
