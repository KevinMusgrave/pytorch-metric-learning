#! /usr/bin/env python3

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import warnings

from . import stat_utils


def precision_at_k(knn_labels, gt_labels, k):
    """
    Precision at k is the percentage of k nearest neighbors that have the correct
    label.
    Args:
        knn_labels: numpy array of size (num_samples, k)
        gt_labels: numpy array of size (num_samples, 1)
    """
    curr_knn_labels = knn_labels[:, :k]
    precision = np.mean(np.sum(curr_knn_labels == gt_labels, axis=1) / k)
    return precision


def mean_average_precision(knn_labels, gt_labels):
    """
    See this for an explanation:
    https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf
    """
    num_samples, num_k = knn_labels.shape
    equality = knn_labels == gt_labels
    num_correct_per_row = np.sum(equality, axis=1)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_per_row = np.sum(precision_at_ks, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_precision_per_row = summed_precision_per_row / num_correct_per_row
    avg_precision_per_row[np.isnan(avg_precision_per_row)] = 0
    return np.mean(avg_precision_per_row)


def recall_at_k(knn_labels, gt_labels, k):
    """
    "Recall at k" in metric learning papers is defined as number of samples
    that have at least 1 correct neighbor out of k.
    """
    num_samples = knn_labels.shape[0]
    curr_knn_labels = knn_labels[:, :k]
    num_matches_per_datapoint = np.sum(curr_knn_labels == gt_labels, axis=1)
    return np.sum(np.clip(num_matches_per_datapoint, 0, 1)) / num_samples


def NMI(input_embeddings, gt_labels):
    """
    Returns NMI and also the predicted labels from k-means
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_clusters = len(set(gt_labels.flatten()))
        pred_labels = stat_utils.run_kmeans(input_embeddings, num_clusters)
        nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    return nmi, pred_labels


def compute_accuracies(query_embeddings, knn_labels, query_labels, k):
    """
    Computes clustering quality of query_embeddings.
    Computes various retrieval scores given knn_labels (labels of nearest neighbors)
    and the ground-truth labels of the query embeddings.
    Returns the results in a dictionary.
    """
    accuracies = {}
    accuracies["mean_average_precision_at_%d"%(k)] = mean_average_precision(knn_labels, query_labels[:, None])
    accuracies["NMI"] = NMI(query_embeddings, query_labels)[0]
    accuracies["precision_at_%d"%(k)] = precision_at_k(knn_labels, query_labels[:, None], k)
    accuracies["recall_at_1"] = recall_at_k(knn_labels, query_labels[:, None], 1)
    return accuracies


def calculate_accuracy(
    query,
    reference,
    query_labels,
    reference_labels,
    k,
    embeddings_come_from_same_source,
):
    """
    Gets k nearest reference embeddings for each element of query.
    Then computes various accuracy metrics.
    """
    embeddings_come_from_same_source = embeddings_come_from_same_source or (
        query is reference
    )
    knn_indices = stat_utils.get_knn(
        reference,
        query,
        k,
        embeddings_come_from_same_source=embeddings_come_from_same_source,
    )
    knn_labels = reference_labels[knn_indices]
    return compute_accuracies(query, knn_labels, query_labels, k)
