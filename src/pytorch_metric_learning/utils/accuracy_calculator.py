#! /usr/bin/env python3

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from . import stat_utils

def get_relevance_mask(shape, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    # This assumes that k was set to at least the max number of relevant items 
    if label_counts is None:
        label_counts = {k:v for k,v in zip(*np.unique(gt_labels, return_counts=True))}
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k,v in label_counts.items():
        matching_rows = np.where(gt_labels==k)[0]
        max_column = v-1 if embeddings_come_from_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask

def r_precision(knn_labels, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeddings_come_from_same_source, label_counts)
    matches_per_row = np.sum((knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1) 
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    return np.mean(matches_per_row / max_possible_matches_per_row)

def mean_average_precision_at_r(knn_labels, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeddings_come_from_same_source, label_counts)
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_per_row = np.sum(precision_at_ks * relevance_mask, axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    return np.mean(summed_precision_per_row / max_possible_matches_per_row)

def precision_at_k(knn_labels, gt_labels, k):
    curr_knn_labels = knn_labels[:, :k]
    precision = np.mean(np.sum(curr_knn_labels == gt_labels, axis=1) / k)
    return precision

def get_label_counts(reference_labels):
    unique_labels, label_counts = np.unique(reference_labels, return_counts=True)
    num_k = min(1023, int(np.max(label_counts))) # faiss can only do a max of k=1024, and we have to do k+1
    return {k:v for k,v in zip(unique_labels, label_counts)}, num_k


class AccuracyCalculator:
    def __init__(self, exclude_metrics=()):
        self.function_keyword = "calculate_"
        function_names = [x for x in dir(self) if x.startswith(self.function_keyword)]
        metrics = [x.replace(self.function_keyword, "", 1) for x in function_names]
        function_dict = {x:getattr(self, y) for x,y in zip(metrics, function_names)}
        self.original_function_dict = {k:v for k,v in function_dict.items() if k not in exclude_metrics}
        self.curr_function_dict = self.get_function_dict()

    def get_function_dict(self, exclude_metrics=()):
        return {k:v for k,v in self.original_function_dict.items() if k not in exclude_metrics}

    def get_curr_metrics(self):
        return [k for k in self.curr_function_dict.keys()]

    def requires_clustering(self):
        return ["NMI", "AMI"]

    def get_cluster_labels(self, query, query_labels, **kwargs):
        num_clusters = len(set(query_labels.flatten()))
        return stat_utils.run_kmeans(query, num_clusters)

    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        return normalized_mutual_info_score(query_labels, cluster_labels)

    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        return adjusted_mutual_info_score(query_labels, cluster_labels)

    def calculate_precision_at_1(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(knn_labels, query_labels[:, None], 1)
        
    def calculate_mean_average_precision_at_r(self, knn_labels, query_labels, embeddings_come_from_same_source=False, label_counts=None, **kwargs):
        return mean_average_precision_at_r(knn_labels, query_labels[:, None], embeddings_come_from_same_source, label_counts)

    def calculate_r_precision(self, knn_labels, query_labels, embeddings_come_from_same_source=False, label_counts=None, **kwargs):
        return r_precision(knn_labels, query_labels[:, None], embeddings_come_from_same_source, label_counts)

    def get_accuracy(self, query, reference, query_labels, reference_labels, embeddings_come_from_same_source, exclude_metrics=()):
        embeddings_come_from_same_source = embeddings_come_from_same_source or (query is reference)
        label_counts, num_k = get_label_counts(reference_labels)

        knn_indices = stat_utils.get_knn(reference, query, num_k, embeddings_come_from_same_source)
        knn_labels = reference_labels[knn_indices]

        kwargs = {"query": query, 
                "reference": reference,
                "query_labels": query_labels,
                "reference_labels": reference_labels,
                "embeddings_come_from_same_source": embeddings_come_from_same_source,
                "label_counts": label_counts,
                "knn_labels": knn_labels}

        self.curr_function_dict = self.get_function_dict(exclude_metrics)

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)                

        return self._get_accuracy(self.curr_function_dict, **kwargs)

    def _get_accuracy(self, function_dict, **kwargs):
        return {k:v(**kwargs) for k,v in function_dict.items()}
