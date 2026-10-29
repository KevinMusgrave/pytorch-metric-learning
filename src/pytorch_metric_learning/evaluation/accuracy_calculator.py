import importlib
import logging
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from ..distances import LpDistance
from ..types import KMeansFunction, KNNFunction, LabelComparison
from ..utils import _common
from .retrieval import CustomKNN

__all__ = ["AccuracyCalculator"]
LOGGER = logging.getLogger(__name__)
DEFAULT_KNN_BATCH_SIZE = 256
KMEANS_REQUIRED = (
    "NMI and AMI require kmeans_func, for example FaissKMeans() from "
    "the faiss extra or a function returning one cluster index per row"
)
type KValue = int | Literal["max_bin_count"] | None
type MetricValue = float | dict[Hashable, float]
type MetricFunction = Callable[..., MetricValue]
type ArrayLike = Tensor | np.ndarray


def mutual_information_scores() -> Any:
    try:
        return importlib.import_module("sklearn.metrics")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "NMI and AMI require scikit-learn. Install the optional dependency with "
            "`pip install pytorch-metric-learning[clustering]`."
        ) from error


def relevant_counts(
    unique_labels: Tensor,
    reference_labels: Tensor,
    ref_includes_query: bool,
    label_comparison_fn: LabelComparison | None,
) -> Tensor:
    if label_comparison_fn is None:
        comparison = unique_labels.unsqueeze(1) == reference_labels.unsqueeze(0)
        counts = _common.all_from_dim(comparison, 2).sum(dim=1)
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


def relevance_mask(counts_per_query: Tensor, num_k: int) -> Tensor:
    positions = torch.arange(num_k, device=counts_per_query.device)
    return positions < counts_per_query.unsqueeze(1)


def same_label(
    knn_labels: Tensor, query_labels: Tensor, label_comparison_fn: LabelComparison
) -> Tensor:
    return label_comparison_fn(query_labels.unsqueeze(1), knn_labels)


def precision_at_k(
    knn_labels: Tensor,
    query_labels: Tensor,
    k: int,
    label_comparison_fn: LabelComparison,
) -> Tensor:
    matches = same_label(knn_labels[:, :k], query_labels, label_comparison_fn)
    return matches.sum(dim=1, dtype=torch.float64) / k


def r_precision(
    knn_labels: Tensor,
    query_labels: Tensor,
    counts_per_query: Tensor,
    label_comparison_fn: LabelComparison,
) -> Tensor:
    mask = relevance_mask(counts_per_query, knn_labels.shape[1])
    matches = same_label(knn_labels, query_labels, label_comparison_fn) & mask
    return matches.sum(dim=1, dtype=torch.float64) / counts_per_query


def mean_average_precision(
    knn_labels: Tensor,
    query_labels: Tensor,
    counts_per_query: Tensor,
    label_comparison_fn: LabelComparison,
    at_r: bool = False,
) -> Tensor:
    num_k = knn_labels.shape[1]
    matches = same_label(knn_labels, query_labels, label_comparison_fn)
    if at_r:
        matches = matches & relevance_mask(counts_per_query, num_k)
    ranks = torch.arange(1, num_k + 1, device=matches.device, dtype=torch.float64)
    precision = torch.cumsum(matches, dim=1, dtype=torch.float64) / ranks
    return (precision * matches).sum(dim=1) / counts_per_query


def mean_reciprocal_rank(
    knn_labels: Tensor, query_labels: Tensor, label_comparison_fn: LabelComparison
) -> Tensor:
    matches = same_label(knn_labels, query_labels, label_comparison_fn)
    if matches.shape[1] == 0:
        return torch.zeros(len(matches), dtype=torch.float64, device=matches.device)
    first_match = torch.argmax(matches.to(torch.uint8), dim=1).to(torch.float64) + 1
    return torch.where(matches.any(dim=1), 1 / first_match, 0.0)


def class_key(label: Tensor) -> Hashable:
    return label.item() if label.ndim == 0 else tuple(label.tolist())


class AccuracyCalculator:
    def __init__(
        self,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        avg_of_avgs: bool = False,
        k: KValue = None,
        label_comparison_fn: LabelComparison | None = None,
        device: torch.device | str | None = None,
        knn_func: KNNFunction | None = None,
        kmeans_func: KMeansFunction | None = None,
    ) -> None:
        self.check_metric_names(include, exclude)
        if len(include) == 0:
            include = [
                x
                for x in self.metric_functions()
                if x not in self.requires_clustering()
            ]
        self.metrics = tuple(x for x in include if x not in exclude)
        self.avg_of_avgs = avg_of_avgs
        self.device = None if device is None else torch.device(device)
        self.knn_func: KNNFunction = (
            CustomKNN(
                LpDistance(normalize_embeddings=False, power=2),
                batch_size=DEFAULT_KNN_BATCH_SIZE,
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
        uses_clustering = any(x in self.requires_clustering() for x in self.metrics)
        if label_comparison_fn is not None and uses_clustering:
            raise ValueError("Clustering metrics require the default label comparison")
        if self.kmeans_func is None and uses_clustering:
            raise ValueError(KMEANS_REQUIRED)

    def metric_functions(self) -> dict[str, MetricFunction]:
        return {
            "AMI": self.calculate_AMI,
            "NMI": self.calculate_NMI,
            "mean_average_precision": self.calculate_mean_average_precision,
            "mean_average_precision_at_r": self.calculate_mean_average_precision_at_r,
            "mean_reciprocal_rank": self.calculate_mean_reciprocal_rank,
            "precision_at_1": self.calculate_precision_at_1,
            "r_precision": self.calculate_r_precision,
        }

    def requires_clustering(self) -> list[str]:
        return ["NMI", "AMI"]

    def requires_knn(self) -> list[str]:
        return [
            "precision_at_1",
            "mean_average_precision",
            "mean_average_precision_at_r",
            "mean_reciprocal_rank",
            "r_precision",
        ]

    def get_cluster_labels(self, query: Tensor, query_labels: Tensor) -> Tensor:
        if self.kmeans_func is None:
            raise ValueError(KMEANS_REQUIRED)
        num_clusters = len(torch.unique(query_labels, dim=0))
        return self.kmeans_func(query, num_clusters)

    def calculate_NMI(
        self, query_labels: Tensor, cluster_labels: Tensor, **kwargs
    ) -> float:
        return float(
            mutual_information_scores().normalized_mutual_info_score(
                *host_arrays(query_labels, cluster_labels)
            )
        )

    def calculate_AMI(
        self, query_labels: Tensor, cluster_labels: Tensor, **kwargs
    ) -> float:
        return float(
            mutual_information_scores().adjusted_mutual_info_score(
                *host_arrays(query_labels, cluster_labels)
            )
        )

    def calculate_precision_at_1(
        self, knn_labels: Tensor, query_labels: Tensor, **kwargs
    ) -> MetricValue:
        per_query = precision_at_k(
            knn_labels, query_labels, 1, self.label_comparison_fn
        )
        return self.aggregate(per_query, **kwargs)

    def calculate_mean_average_precision_at_r(
        self,
        knn_labels: Tensor,
        query_labels: Tensor,
        relevant_counts: Tensor,
        **kwargs,
    ) -> MetricValue:
        per_query = mean_average_precision(
            knn_labels,
            query_labels,
            relevant_counts,
            self.label_comparison_fn,
            at_r=True,
        )
        return self.aggregate(per_query, **kwargs)

    def calculate_mean_average_precision(
        self,
        knn_labels: Tensor,
        query_labels: Tensor,
        relevant_counts: Tensor,
        **kwargs,
    ) -> MetricValue:
        per_query = mean_average_precision(
            knn_labels, query_labels, relevant_counts, self.label_comparison_fn
        )
        return self.aggregate(per_query, **kwargs)

    def calculate_mean_reciprocal_rank(
        self, knn_labels: Tensor, query_labels: Tensor, **kwargs
    ) -> MetricValue:
        per_query = mean_reciprocal_rank(
            knn_labels, query_labels, self.label_comparison_fn
        )
        return self.aggregate(per_query, **kwargs)

    def calculate_r_precision(
        self,
        knn_labels: Tensor,
        query_labels: Tensor,
        relevant_counts: Tensor,
        **kwargs,
    ) -> MetricValue:
        per_query = r_precision(
            knn_labels, query_labels, relevant_counts, self.label_comparison_fn
        )
        return self.aggregate(per_query, **kwargs)

    def aggregate(
        self,
        per_query: Tensor,
        not_lone_query_mask: Tensor,
        unique_query_labels: Tensor,
        query_label_index: Tensor,
        per_class: bool = False,
        **kwargs,
    ) -> MetricValue:
        values = per_query.to(torch.float64)[not_lone_query_mask]
        if not (self.avg_of_avgs or per_class):
            return values.mean().item() if len(values) else float("nan")
        classes = query_label_index[not_lone_query_mask]
        num_classes = len(unique_query_labels)
        sums = values.new_zeros(num_classes).index_add(0, classes, values)
        counts = torch.bincount(classes, minlength=num_classes)
        class_means = torch.where(counts > 0, sums / counts.clamp_min(1), torch.nan)
        if per_class:
            return {
                class_key(label): value
                for label, value in zip(
                    unique_query_labels.cpu(), class_means.cpu().tolist(), strict=True
                )
            }
        present = class_means[counts > 0]
        return present.mean().item() if len(present) else float("nan")

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
        results = self.compute_metrics(
            query,
            query_labels,
            reference,
            reference_labels,
            ref_includes_query,
            self.selected_metrics(include, exclude),
            per_class=False,
        )
        return cast(dict[str, float], results)

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
        names = [
            name
            for name in self.selected_metrics(include, exclude)
            if name in self.requires_knn()
        ]
        results = self.compute_metrics(
            query,
            query_labels,
            reference,
            reference_labels,
            ref_includes_query,
            names,
            per_class=True,
        )
        return cast(dict[str, dict[Hashable, float]], results)

    def selected_metrics(
        self, include: Sequence[str], exclude: Sequence[str]
    ) -> list[str]:
        self.check_metric_names(include, exclude)
        if not set(include) <= set(self.metrics):
            raise ValueError(
                f"include must name metrics selected at construction: {self.metrics}"
            )
        return [name for name in (include or self.metrics) if name not in exclude]

    def compute_metrics(
        self,
        query: ArrayLike,
        query_labels: ArrayLike,
        reference: ArrayLike | None,
        reference_labels: ArrayLike | None,
        ref_includes_query: bool,
        names: Sequence[str],
        per_class: bool,
    ) -> dict[str, MetricValue]:
        if reference is None:
            reference = query
            reference_labels = query_labels
            ref_includes_query = True
        if reference_labels is None:
            raise ValueError("reference_labels are required with reference")

        device = self.device
        if device is None:
            device = torch.as_tensor(query).device
        query_t, reference_t = embedding_pair(query, reference, device)
        query_labels_t, reference_labels_t = (
            torch.as_tensor(x).to(device) for x in (query_labels, reference_labels)
        )

        if len(query_t) != len(query_labels_t) or len(reference_t) != len(
            reference_labels_t
        ):
            raise ValueError("embeddings and labels must have the same length")
        if not (torch.isfinite(query_t).all() and torch.isfinite(reference_t).all()):
            raise ValueError("query and reference embeddings must be finite")
        if ref_includes_query and not (
            torch.allclose(query_t, reference_t[: len(query_t)])
            and labels_equal(query_labels_t, reference_labels_t[: len(query_labels_t)])
        ):
            raise ValueError(
                "When ref_includes_query is True, the first len(query) rows of "
                "reference and reference_labels must equal query and query_labels."
            )

        functions = {name: self.metric_functions()[name] for name in names}
        kwargs: dict[str, Any] = {
            "query": query_t,
            "reference": reference_t,
            "query_labels": query_labels_t,
            "reference_labels": reference_labels_t,
            "ref_includes_query": ref_includes_query,
            "label_comparison_fn": self.label_comparison_fn,
            "per_class": per_class,
        }
        if any(x in self.requires_knn() for x in names):
            kwargs.update(
                self.knn_inputs(
                    query_t,
                    query_labels_t,
                    reference_t,
                    reference_labels_t,
                    ref_includes_query,
                    names,
                )
            )
        if any(x in self.requires_clustering() for x in names):
            kwargs["cluster_labels"] = self.get_cluster_labels(query_t, query_labels_t)
        return {name: function(**kwargs) for name, function in functions.items()}

    def knn_inputs(
        self,
        query: Tensor,
        query_labels: Tensor,
        reference: Tensor,
        reference_labels: Tensor,
        ref_includes_query: bool,
        names: Sequence[str],
    ) -> dict[str, Tensor]:
        unique_labels, label_index = torch.unique(
            query_labels, dim=0, return_inverse=True
        )
        counts = relevant_counts(
            unique_labels, reference_labels, ref_includes_query, self.custom_comparison
        )
        counts_per_query = counts[label_index]
        not_lone = counts_per_query > 0
        if not torch.any(not_lone):
            LOGGER.warning("None of the query labels are in the reference set.")
        num_k = self.determine_k(counts, len(reference), ref_includes_query, names)
        knn_distances, knn_indices = self.knn_func(
            query, num_k, reference, ref_includes_query
        )
        return {
            "knn_labels": reference_labels[knn_indices],
            "knn_distances": knn_distances,
            "relevant_counts": counts_per_query,
            "not_lone_query_mask": not_lone,
            "unique_query_labels": unique_labels,
            "query_label_index": label_index,
        }

    def check_metric_names(
        self, include: Sequence[str], exclude: Sequence[str]
    ) -> None:
        names = list(self.metric_functions())
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
        names: Sequence[str],
    ) -> int:
        available = num_reference_embeddings - int(ref_includes_query)
        max_relevant = int(counts.max().item()) if len(counts) else 0
        if self.k == "max_bin_count":
            return max(min(max_relevant, available), 1)
        if self.k is None:
            return available
        at_r = set(names) & {"r_precision", "mean_average_precision_at_r"}
        if self.k < max_relevant and at_r:
            LOGGER.warning(
                "You are computing %s, but k (%s) is less than the largest number "
                "of relevant references (%s), so these metrics will be incorrect. "
                "Set k='max_bin_count'. For MAP@%s instead of MAP@R, use "
                "'mean_average_precision' rather than 'mean_average_precision_at_r'.",
                sorted(at_r),
                self.k,
                max_relevant,
                self.k,
            )
        return min(self.k, available)


def embedding_pair(
    query: ArrayLike, reference: ArrayLike, device: torch.device
) -> tuple[Tensor, Tensor]:
    query_t, reference_t = (torch.as_tensor(x).to(device) for x in (query, reference))
    if not (query_t.is_floating_point() and reference_t.is_floating_point()):
        return query_t.float(), reference_t.float()
    dtype = torch.promote_types(query_t.dtype, reference_t.dtype)
    return query_t.to(dtype), reference_t.to(dtype)


def host_arrays(*values: ArrayLike) -> list[np.ndarray]:
    return [torch.as_tensor(value).cpu().numpy() for value in values]


def labels_equal(x: Tensor, y: Tensor) -> bool:
    if x.is_floating_point() or y.is_floating_point():
        dtype = torch.promote_types(x.dtype, y.dtype)
        return torch.allclose(x.to(dtype), y.to(dtype))
    return torch.equal(x, y)
