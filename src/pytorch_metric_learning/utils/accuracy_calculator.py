#! /usr/bin/env python3

import logging

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from . import stat_utils


def maybe_get_avg_of_avgs(accuracy_per_sample, sample_labels, avg_of_avgs):
    if avg_of_avgs:
        unique_labels = np.unique(sample_labels)
        mask = sample_labels == unique_labels[None, :]
        acc_sum_per_class = np.sum(accuracy_per_sample[:, None] * mask, axis=0)
        mask_sum_per_class = np.sum(mask, axis=0)
        average_per_class = acc_sum_per_class / mask_sum_per_class
        return np.mean(average_per_class)
    return np.mean(accuracy_per_sample)


def get_relevance_mask(
    shape, gt_labels, embeddings_come_from_same_source, label_counts
):
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, v in label_counts.items():
        matching_rows = np.where(gt_labels == k)[0]
        max_column = v - 1 if embeddings_come_from_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask


def r_precision(
    knn_labels, gt_labels, embeddings_come_from_same_source, label_counts, avg_of_avgs
):
    relevance_mask = get_relevance_mask(
        knn_labels.shape, gt_labels, embeddings_come_from_same_source, label_counts
    )
    matches_per_row = np.sum(
        (knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1
    )
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = matches_per_row / max_possible_matches_per_row
    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def mean_average_precision(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    avg_of_avgs,
    relevance_mask=None,
    at_r=False,
):
    relevance_mask = (
        np.ones_like(knn_labels) if relevance_mask is None else relevance_mask
    )
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_per_row = np.sum(precision_at_ks * relevance_mask, axis=1)
    if at_r:
        max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    else:
        max_possible_matches_per_row = np.sum(equality, axis=1)
        max_possible_matches_per_row[max_possible_matches_per_row == 0] = 1
    accuracy_per_sample = summed_precision_per_row / max_possible_matches_per_row
    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def mean_average_precision_at_r(
    knn_labels, gt_labels, embeddings_come_from_same_source, label_counts, avg_of_avgs
):
    relevance_mask = get_relevance_mask(
        knn_labels.shape, gt_labels, embeddings_come_from_same_source, label_counts
    )
    return mean_average_precision(
        knn_labels,
        gt_labels,
        embeddings_come_from_same_source,
        avg_of_avgs,
        relevance_mask,
        at_r=True,
    )


def precision_at_k(knn_labels, gt_labels, k, avg_of_avgs):
    curr_knn_labels = knn_labels[:, :k]
    accuracy_per_sample = np.sum(curr_knn_labels == gt_labels, axis=1) / k
    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def get_label_counts(reference_labels):
    unique_labels, label_counts = np.unique(reference_labels, return_counts=True)
    num_k = min(
        1023, int(np.max(label_counts))
    )  # faiss can only do a max of k=1024, and we have to do k+1
    return {k: v for k, v in zip(unique_labels, label_counts)}, num_k


def get_lone_query_labels(
    query_labels,
    reference_labels,
    reference_label_counts,
    embeddings_come_from_same_source,
):
    if embeddings_come_from_same_source:
        return np.array([k for k, v in reference_label_counts.items() if v <= 1])
    else:
        return np.setdiff1d(query_labels, reference_labels)


def try_getting_not_lone_labels(knn_labels, query_labels, not_lone_query_mask):
    if not any(not_lone_query_mask):
        return None, None
    return (
        knn_labels[not_lone_query_mask],
        query_labels[not_lone_query_mask],
    )


class AccuracyCalculator:
    def __init__(self, include=(), exclude=(), avg_of_avgs=False, k=None):
        self.function_keyword = "calculate_"
        function_names = [x for x in dir(self) if x.startswith(self.function_keyword)]
        metrics = [x.replace(self.function_keyword, "", 1) for x in function_names]
        self.original_function_dict = {
            x: getattr(self, y) for x, y in zip(metrics, function_names)
        }
        self.check_primary_metrics(include, exclude)
        self.original_function_dict = self.get_function_dict(include, exclude)
        self.curr_function_dict = self.get_function_dict()
        self.avg_of_avgs = avg_of_avgs
        self.k = k

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
        return [k for k in self.curr_function_dict.keys()]

    def requires_clustering(self):
        return ["NMI", "AMI"]

    def requires_knn(self):
        return [
            "precision_at_1",
            "mean_average_precision",
            "mean_average_precision_at_r",
            "r_precision",
        ]

    def get_cluster_labels(self, query, query_labels, **kwargs):
        num_clusters = len(np.unique(query_labels.flatten()))
        return stat_utils.run_kmeans(query, num_clusters)

    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        return normalized_mutual_info_score(query_labels, cluster_labels)

    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        return adjusted_mutual_info_score(query_labels, cluster_labels)

    def calculate_precision_at_1(
        self, knn_labels, query_labels, not_lone_query_mask, **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return precision_at_k(knn_labels, query_labels[:, None], 1, self.avg_of_avgs)

    def calculate_mean_average_precision_at_r(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return mean_average_precision_at_r(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
        )

    def calculate_mean_average_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return mean_average_precision(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            self.avg_of_avgs,
        )

    def calculate_r_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return r_precision(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
        )

    def get_accuracy(
        self,
        query,
        reference,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source,
        include=(),
        exclude=(),
    ):
        embeddings_come_from_same_source = embeddings_come_from_same_source or (
            query is reference
        )

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "embeddings_come_from_same_source": embeddings_come_from_same_source,
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts, num_k = get_label_counts(reference_labels)
            if self.k is not None:
                num_k = self.k
            knn_indices, knn_distances = stat_utils.get_knn(
                reference, query, num_k, embeddings_come_from_same_source
            )
            knn_labels = reference_labels[knn_indices]
            lone_query_labels = get_lone_query_labels(
                query_labels,
                reference_labels,
                label_counts,
                embeddings_come_from_same_source,
            )
            not_lone_query_mask = ~np.isin(query_labels, lone_query_labels)
            if not any(not_lone_query_mask):
                logging.warning("None of the query labels are in the reference set.")
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

    def check_primary_metrics(calc, include=(), exclude=()):
        primary_metrics = list(calc.original_function_dict.keys())
        for met in [include, exclude]:
            if not isinstance(met, (tuple, list)):
                raise TypeError(
                    "Arguments must be of type tuple, not {}.".format(type(met))
                )
            if not set(met).issubset(set(primary_metrics)):
                raise ValueError(
                    "Primary metrics must be one or more of: {}.".format(
                        primary_metrics
                    )
                )

    def description(self):
        return "avg_of_avgs" if self.avg_of_avgs else ""
