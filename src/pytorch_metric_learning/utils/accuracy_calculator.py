import importlib
import logging

import torch

from ..distances import LpDistance
from . import common_functions as c_f
from .retrieval import CustomKNN

EQUALITY = torch.eq
LOGGER = logging.getLogger(__name__)
DEFAULT_KNN_BATCH_SIZE = 256
KMEANS_REQUIRED = (
    "NMI and AMI require kmeans_func, for example FaissKMeans() from "
    "the faiss extra or a function returning one cluster index per row"
)


def mutual_information_scores():
    try:
        return importlib.import_module("sklearn.metrics")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "NMI and AMI require scikit-learn. Install the optional dependency with "
            "`pip install pytorch-metric-learning[clustering]`."
        ) from error


def get_unique_labels(labels):
    return torch.unique(labels, dim=0)


def maybe_get_avg_of_avgs(
    accuracy_per_sample, sample_labels, avg_of_avgs, return_per_class
):
    if avg_of_avgs or return_per_class:
        unique_labels = get_unique_labels(sample_labels)
        mask = c_f.torch_all_from_dim_to_end(
            sample_labels == unique_labels.unsqueeze(1), 2
        )
        mask = torch.t(mask)
        acc_sum_per_class = torch.sum(accuracy_per_sample.unsqueeze(1) * mask, dim=0)
        mask_sum_per_class = torch.sum(mask, dim=0)
        average_per_class = acc_sum_per_class / mask_sum_per_class
        if return_per_class:
            return average_per_class.cpu().tolist()
        return torch.mean(average_per_class).item()
    return torch.mean(accuracy_per_sample).item()


def get_relevance_mask(
    shape,
    gt_labels,
    ref_includes_query,
    label_counts,
):
    relevance_mask = torch.zeros(size=shape, dtype=torch.bool, device=gt_labels.device)
    count_per_query = torch.zeros(
        len(gt_labels), dtype=torch.long, device=gt_labels.device
    )

    for label, count in zip(*label_counts, strict=True):
        matching_rows = torch.where(
            c_f.torch_all_from_dim_to_end(gt_labels == label, 1)
        )[0]
        max_column = count - 1 if ref_includes_query else count
        relevance_mask[matching_rows, :max_column] = True
        count_per_query[matching_rows] = max_column
    return relevance_mask, count_per_query


def r_precision(
    knn_labels,
    gt_labels,
    ref_includes_query,
    label_counts,
    avg_of_avgs,
    return_per_class,
    label_comparison_fn,
):
    relevance_mask, _ = get_relevance_mask(
        knn_labels.shape[:2],
        gt_labels,
        ref_includes_query,
        label_counts,
    )
    same_label = label_comparison_fn(gt_labels, knn_labels)
    matches_per_row = torch.sum(same_label * relevance_mask, dim=1)
    max_possible_matches_per_row = torch.sum(relevance_mask, dim=1)
    accuracy_per_sample = (
        matches_per_row.type(torch.float64) / max_possible_matches_per_row
    )
    return maybe_get_avg_of_avgs(
        accuracy_per_sample, gt_labels, avg_of_avgs, return_per_class
    )


def mean_average_precision(
    knn_labels,
    gt_labels,
    ref_includes_query,
    label_counts,
    avg_of_avgs,
    return_per_class,
    label_comparison_fn,
    at_r=False,
):
    device = gt_labels.device
    num_samples, num_k = knn_labels.shape[:2]
    relevance_mask, count_per_query = get_relevance_mask(
        knn_labels.shape[:2],
        gt_labels,
        ref_includes_query,
        label_counts,
    )
    knn_mask = (
        relevance_mask
        if at_r
        else torch.ones((num_samples, num_k), dtype=torch.bool, device=device)
    )
    is_same_label = label_comparison_fn(gt_labels, knn_labels)
    equality = is_same_label * knn_mask
    cumulative_correct = torch.cumsum(equality, dim=1)
    k_idx = torch.arange(1, num_k + 1, device=device).repeat(num_samples, 1)
    precision_at_ks = (cumulative_correct * equality).type(torch.float64) / k_idx
    summed_precision_per_row = torch.sum(precision_at_ks * knn_mask, dim=1)
    accuracy_per_sample = summed_precision_per_row / count_per_query
    return maybe_get_avg_of_avgs(
        accuracy_per_sample, gt_labels, avg_of_avgs, return_per_class
    )


def mean_reciprocal_rank(
    knn_labels,
    gt_labels,
    avg_of_avgs,
    return_per_class,
    label_comparison_fn,
):
    device = gt_labels.device
    is_same_label = label_comparison_fn(gt_labels, knn_labels)

    sum_per_row = is_same_label.sum(-1)
    zero_remove_mask = sum_per_row > 0
    indices = torch.arange(is_same_label.shape[1], 0, -1, device=device)
    tmp = is_same_label * indices
    indices = torch.argmax(tmp, 1, keepdim=True) + 1.0

    indices[zero_remove_mask] = 1.0 / indices[zero_remove_mask]
    indices[~zero_remove_mask] = 0.0

    indices = indices.flatten()

    return maybe_get_avg_of_avgs(indices, gt_labels, avg_of_avgs, return_per_class)


def precision_at_k(
    knn_labels, gt_labels, k, avg_of_avgs, return_per_class, label_comparison_fn
):
    curr_knn_labels = knn_labels[:, :k]
    same_label = label_comparison_fn(gt_labels, curr_knn_labels)
    accuracy_per_sample = torch.sum(same_label, dim=1).type(torch.float64) / k
    return maybe_get_avg_of_avgs(
        accuracy_per_sample, gt_labels, avg_of_avgs, return_per_class
    )


def get_label_match_counts(query_labels, reference_labels, label_comparison_fn):
    unique_query_labels = get_unique_labels(query_labels)
    if label_comparison_fn is EQUALITY:
        comparison = unique_query_labels[:, None] == reference_labels
        match_counts = torch.sum(c_f.torch_all_from_dim_to_end(comparison, 2), dim=1)
    else:
        match_counts = torch.empty(
            len(unique_query_labels), dtype=torch.long, device=query_labels.device
        )
        for ix_a in range(len(unique_query_labels)):
            label_a = unique_query_labels[ix_a : ix_a + 1]
            match_counts[ix_a] = torch.sum(
                label_comparison_fn(label_a, reference_labels)
            )

    return (unique_query_labels, match_counts)


def get_lone_query_labels(
    query_labels,
    label_counts,
    ref_includes_query,
    label_comparison_fn,
):
    unique_labels, match_counts = label_counts
    if ref_includes_query:
        label_matches_itself = label_comparison_fn(unique_labels, unique_labels)
        lone_condition = match_counts - label_matches_itself.type(torch.long) <= 0
    else:
        lone_condition = match_counts == 0
    lone_query_labels = unique_labels[lone_condition]
    if len(lone_query_labels) > 0:
        comparison = query_labels[:, None] == lone_query_labels
        not_lone_query_mask = ~torch.any(
            c_f.torch_all_from_dim_to_end(comparison, 2), dim=1
        )
    else:
        not_lone_query_mask = torch.ones(
            query_labels.shape[0], dtype=torch.bool, device=query_labels.device
        )
    return lone_query_labels, not_lone_query_mask


def try_getting_not_lone_labels(knn_labels, query_labels, not_lone_query_mask):
    if not any(not_lone_query_mask):
        return None, None
    return (
        knn_labels[not_lone_query_mask],
        query_labels[not_lone_query_mask],
    )


def nan_accuracy(unique_labels, return_per_class):
    if return_per_class:
        return [float("nan") for _ in range(len(unique_labels))]
    return float("nan")


class AccuracyCalculator:
    def __init__(
        self,
        include=(),
        exclude=(),
        avg_of_avgs=False,
        return_per_class=False,
        k=None,
        label_comparison_fn=None,
        device=None,
        knn_func=None,
        kmeans_func=None,
    ):
        self.original_function_dict = self.metric_functions()
        self.check_primary_metrics(include, exclude)
        if len(include) == 0:
            include = [
                x
                for x in self.original_function_dict
                if x not in self.requires_clustering()
            ]
        self.original_function_dict = self.get_function_dict(include, exclude)
        self.curr_function_dict = self.get_function_dict()

        if avg_of_avgs and return_per_class:
            raise ValueError("avg_of_avgs and return_per_class are mutually exclusive")
        self.avg_of_avgs = avg_of_avgs
        self.return_per_class = return_per_class

        self.device = None if device is None else torch.device(device)
        self.knn_func = (
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

        if label_comparison_fn:
            self.label_comparison_fn = label_comparison_fn
            if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
                raise NotImplementedError(
                    "Unsupported: clustering + custom label comparison"
                )
        else:
            self.label_comparison_fn = EQUALITY

        if self.kmeans_func is None and any(
            x in self.requires_clustering() for x in self.get_curr_metrics()
        ):
            raise ValueError(KMEANS_REQUIRED)

    def metric_functions(self):
        return {
            "AMI": self.calculate_AMI,
            "NMI": self.calculate_NMI,
            "mean_average_precision": self.calculate_mean_average_precision,
            "mean_average_precision_at_r": (self.calculate_mean_average_precision_at_r),
            "mean_reciprocal_rank": self.calculate_mean_reciprocal_rank,
            "precision_at_1": self.calculate_precision_at_1,
            "r_precision": self.calculate_r_precision,
        }

    def get_function_dict(self, include=(), exclude=()):
        if len(include) == 0:
            include = list(self.original_function_dict.keys())
        included_metrics = [k for k in include if k not in exclude]
        return {
            k: v
            for k, v in self.original_function_dict.items()
            if k in included_metrics
        }

    def get_curr_metrics(self):
        return list(self.curr_function_dict)

    def requires_clustering(self):
        return ["NMI", "AMI"]

    def requires_knn(self):
        return [
            "precision_at_1",
            "mean_average_precision",
            "mean_average_precision_at_r",
            "mean_reciprocal_rank",
            "r_precision",
        ]

    def get_cluster_labels(self, query, query_labels, **kwargs):
        if self.kmeans_func is None:
            raise ValueError(KMEANS_REQUIRED)
        num_clusters = len(torch.unique(query_labels.flatten()))
        return self.kmeans_func(query, num_clusters)

    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        [query_labels, cluster_labels] = [
            torch.as_tensor(x).cpu().numpy() for x in [query_labels, cluster_labels]
        ]
        return mutual_information_scores().normalized_mutual_info_score(
            query_labels, cluster_labels
        )

    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        [query_labels, cluster_labels] = [
            torch.as_tensor(x).cpu().numpy() for x in [query_labels, cluster_labels]
        ]
        return mutual_information_scores().adjusted_mutual_info_score(
            query_labels, cluster_labels
        )

    def calculate_precision_at_1(
        self, knn_labels, query_labels, not_lone_query_mask, label_counts, **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return nan_accuracy(label_counts[0], self.return_per_class)
        return precision_at_k(
            knn_labels,
            query_labels[:, None],
            1,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def calculate_mean_average_precision_at_r(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        ref_includes_query,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return nan_accuracy(label_counts[0], self.return_per_class)
        return mean_average_precision(
            knn_labels,
            query_labels[:, None],
            ref_includes_query,
            label_counts,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
            at_r=True,
        )

    def calculate_mean_average_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        ref_includes_query,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return nan_accuracy(label_counts[0], self.return_per_class)

        return mean_average_precision(
            knn_labels,
            query_labels[:, None],
            ref_includes_query,
            label_counts,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def calculate_mean_reciprocal_rank(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return nan_accuracy(label_counts[0], self.return_per_class)

        return mean_reciprocal_rank(
            knn_labels,
            query_labels[:, None],
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def calculate_r_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        ref_includes_query,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return nan_accuracy(label_counts[0], self.return_per_class)
        return r_precision(
            knn_labels,
            query_labels[:, None],
            ref_includes_query,
            label_counts,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def get_accuracy(
        self,
        query,
        query_labels,
        reference=None,
        reference_labels=None,
        ref_includes_query=False,
        include=(),
        exclude=(),
    ):
        if reference is None:
            reference = query
            reference_labels = query_labels
            ref_includes_query = True

        device = self.device
        if device is None:
            device = torch.as_tensor(query).device
        [query, reference] = [
            torch.as_tensor(x).to(device).type(torch.float)
            for x in [query, reference]
        ]
        [query_labels, reference_labels] = [
            torch.as_tensor(x).to(device) for x in [query_labels, reference_labels]
        ]

        if len(query) != len(query_labels) or len(reference) != len(reference_labels):
            raise ValueError("embeddings and labels must have the same length")

        if not (torch.isfinite(query).all() and torch.isfinite(reference).all()):
            raise ValueError("query and reference embeddings must be finite")

        labels_match = True
        if ref_includes_query:
            reference_prefix = reference_labels[: len(query_labels)]
            if query_labels.is_floating_point() or reference_labels.is_floating_point():
                dtype = torch.promote_types(query_labels.dtype, reference_labels.dtype)
                labels_match = torch.allclose(
                    query_labels.to(dtype), reference_prefix.to(dtype)
                )
            else:
                labels_match = torch.equal(query_labels, reference_prefix)

        if ref_includes_query and not (
            torch.allclose(query, reference[: len(query)]) and labels_match
        ):
            raise ValueError(
                "When ref_includes_query is True, the first len(query) rows of "
                "reference and reference_labels must equal query and query_labels."
            )

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "ref_includes_query": ref_includes_query,
            "label_comparison_fn": self.label_comparison_fn,
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts = get_label_match_counts(
                query_labels, reference_labels, self.label_comparison_fn
            )
            lone_query_labels, not_lone_query_mask = get_lone_query_labels(
                query_labels,
                label_counts,
                ref_includes_query,
                self.label_comparison_fn,
            )

            num_k = self.determine_k(
                label_counts[1], len(reference), ref_includes_query
            )

            knn_distances, knn_indices = self.knn_func(
                query, num_k, reference, ref_includes_query
            )

            knn_labels = reference_labels[knn_indices]
            if not any(not_lone_query_mask):
                LOGGER.warning("None of the query labels are in the reference set.")
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        return self._get_accuracy(self.curr_function_dict, **kwargs)

    def _get_accuracy(self, function_dict, **kwargs):
        return {k: v(**kwargs) for k, v in function_dict.items()}

    def check_primary_metrics(self, include=(), exclude=()):
        primary_metrics = list(self.original_function_dict.keys())
        for met in [include, exclude]:
            if not isinstance(met, (tuple, list)):
                raise TypeError(f"Arguments must be of type tuple, not {type(met)}.")
            if not set(met).issubset(set(primary_metrics)):
                raise ValueError(
                    f"Primary metrics must be one or more of: {primary_metrics}."
                )

    def determine_k(self, bin_counts, num_reference_embeddings, ref_includes_query):
        self_count = int(ref_includes_query)
        max_bin_count = torch.max(bin_counts).item()
        if self.k == "max_bin_count":
            return max_bin_count - self_count
        if self.k is None:
            return num_reference_embeddings - self_count
        if self.k < max_bin_count:
            intersection = set(self.get_curr_metrics()).intersection(
                set({"r_precision", "mean_average_precision_at_r"})
            )
            if len(intersection) > 0:
                LOGGER.warning(
                    "You are computing %s, but k (%s) is less than the max bin "
                    "count (%s), so these metrics will be incorrect. Set "
                    "k='max_bin_count'. For MAP@%s instead of MAP@R, use "
                    "'mean_average_precision' rather than "
                    "'mean_average_precision_at_r'.",
                    intersection,
                    self.k,
                    max_bin_count,
                    self.k,
                )
        return min(self.k, num_reference_embeddings - self_count)
