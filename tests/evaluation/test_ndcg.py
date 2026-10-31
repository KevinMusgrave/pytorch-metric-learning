import math
import unittest
from functools import partial

import torch

from pytorch_metric_learning.evaluation import (
    AccuracyCalculator,
    RetrievalInputs,
    RetrievalMetric,
    ndcg,
)

from .. import TEST_DEVICE


def labels(*values):
    return torch.tensor(values, device=TEST_DEVICE)


def same_class(x, y):
    return x[..., 0] == y[..., 0]


def graded(x, y):
    same_parent = (x[..., 1] == y[..., 1]).to(torch.float64)
    return same_parent + same_class(x, y).to(torch.float64)


def retrieval_inputs(query_labels, knn_labels, reference_labels, self_search=False):
    return RetrievalInputs(
        query_labels=query_labels,
        knn_labels=knn_labels,
        matches=same_class(query_labels.unsqueeze(1), knn_labels),
        relevant_counts=torch.zeros(len(query_labels), device=TEST_DEVICE),
        reference_labels=reference_labels,
        ref_includes_query=self_search,
    )


def dcg(gains):
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))


def equal_relevance(x, y):
    return (x == y).to(torch.float64)


class TestNDCG(unittest.TestCase):
    def test_hand_computed_graded_value(self):
        query = labels([0, 0])
        reference = labels([1, 0], [5, 1], [0, 0], [6, 1], [2, 0])
        knn = reference[[4, 1, 0]].unsqueeze(0)
        value = ndcg(retrieval_inputs(query, knn, reference), graded, k=3)
        self.assertAlmostEqual(value.item(), dcg([1, 0, 1]) / dcg([2, 1, 1]))

    def test_ideal_ranking_uses_unretrieved_references(self):
        query = labels(0)
        reference = labels(0, 0, 0, 1, 1)
        knn = labels(1, 1).unsqueeze(0)
        value = ndcg(retrieval_inputs(query, knn, reference), equal_relevance)
        self.assertEqual(value.item(), 0.0)
        knn = labels(0, 1).unsqueeze(0)
        value = ndcg(retrieval_inputs(query, knn, reference), equal_relevance)
        self.assertAlmostEqual(value.item(), dcg([1, 0]) / dcg([1, 1]))

    def test_perfect_ranking_scores_one_at_every_depth(self):
        query = labels(0)
        reference = labels(0, 0, 1, 1)
        knn = labels(0, 0, 1, 1).unsqueeze(0)
        inputs = retrieval_inputs(query, knn, reference)
        for k in (1, 2, 3, 4, 9, None):
            with self.subTest(k=k):
                self.assertAlmostEqual(ndcg(inputs, equal_relevance, k=k).item(), 1.0)

    def test_depth_is_applied_to_gain_and_ideal(self):
        query = labels(0)
        reference = labels(0, 0, 1)
        knn = labels(1, 0, 0).unsqueeze(0)
        inputs = retrieval_inputs(query, knn, reference)
        self.assertAlmostEqual(
            ndcg(inputs, equal_relevance, k=2).item(), dcg([0, 1]) / dcg([1, 1])
        )

    def test_self_search_excludes_own_row_from_ideal(self):
        query = labels(0, 0, 1)
        reference = labels(0, 0, 1, 1)
        knn = labels([0, 1], [1, 1], [1, 0])
        values = ndcg(
            retrieval_inputs(query, knn, reference, True), equal_relevance
        ).tolist()
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], 0.0)
        self.assertAlmostEqual(values[2], 1.0)

    def test_comparator_that_rejects_self_keeps_ideal_gains(self):
        def different(x, y):
            return (x != y).to(torch.float64)

        query = labels(0)
        reference = labels(0, 1, 1)
        knn = labels(1, 1).unsqueeze(0)
        value = ndcg(retrieval_inputs(query, knn, reference, True), different)
        self.assertAlmostEqual(value.item(), 1.0)

    def test_no_gain_is_undefined(self):
        value = ndcg(
            retrieval_inputs(labels(0), labels(1).unsqueeze(0), labels(1)),
            equal_relevance,
        )
        self.assertTrue(value.isnan().item())

    def test_invalid_gains(self):
        inputs = retrieval_inputs(labels(0), labels(1).unsqueeze(0), labels(1))
        for bad in (-1.0, torch.nan, torch.inf):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "finite, nonnegative"),
            ):
                ndcg(inputs, lambda x, y, bad=bad: torch.full_like(y.double(), bad))

    def test_matches_brute_force_through_calculator(self):
        generator = torch.Generator().manual_seed(2)
        reference = torch.randn(40, 3, generator=generator).to(TEST_DEVICE)
        classes = torch.randint(0, 6, (40,), generator=generator)
        reference_labels = torch.stack([classes, classes // 2], dim=1).to(TEST_DEVICE)
        query = torch.randn(25, 3, generator=generator).to(TEST_DEVICE)
        classes = torch.randint(0, 6, (25,), generator=generator)
        query_labels = torch.stack([classes, classes // 2], dim=1).to(TEST_DEVICE)
        calculator = AccuracyCalculator(
            include=("ndcg_at_5",),
            label_comparison_fn=lambda x, y: graded(x, y) > 0,
            custom_metrics={
                "ndcg_at_5": RetrievalMetric(
                    partial(ndcg, relevance_fn=graded, k=5), neighbors=5
                )
            },
        )
        result = calculator.get_accuracy(
            query, query_labels, reference, reference_labels
        )
        expected = []
        for row, label in zip(query, query_labels, strict=True):
            order = torch.cdist(row[None], reference)[0].argsort()[:5]
            gains = graded(label[None], reference_labels).tolist()
            ranked = [gains[i] for i in order.tolist()]
            ideal = dcg(sorted(gains, reverse=True)[:5])
            if ideal > 0:
                expected.append(dcg(ranked) / ideal)
        self.assertAlmostEqual(result["ndcg_at_5"], sum(expected) / len(expected))

    def test_self_search_through_calculator_ignores_own_row(self):
        embeddings = torch.tensor([[0.0], [1.0], [5.0], [6.0]], device=TEST_DEVICE)
        values = labels(0, 1, 0, 1)
        calculator = AccuracyCalculator(
            include=("ndcg", "precision_at_1"),
            custom_metrics={
                "ndcg": RetrievalMetric(partial(ndcg, relevance_fn=equal_relevance))
            },
        )
        result = calculator.get_accuracy(embeddings, values)
        expected = (dcg([0, 1, 0]) + dcg([0, 0, 1])) / (2 * dcg([1, 0, 0]))
        self.assertAlmostEqual(result["precision_at_1"], 0.0)
        self.assertAlmostEqual(result["ndcg"], expected)

    def test_calculator_bound_shortens_gain_and_ideal(self):
        embeddings = torch.tensor([[0.0], [1.0], [2.0], [3.0]], device=TEST_DEVICE)
        values = labels(0, 1, 0, 0)
        calculator = AccuracyCalculator(
            include=("ndcg",),
            k=2,
            custom_metrics={
                "ndcg": RetrievalMetric(partial(ndcg, relevance_fn=equal_relevance))
            },
        )
        result = calculator.get_accuracy(
            embeddings[:1], values[:1], embeddings, values, True
        )
        self.assertAlmostEqual(result["ndcg"], dcg([0, 1]) / dcg([1, 1]))

    def test_mean_average_precision_uses_binary_labels_alongside_ndcg(self):
        embeddings = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [9.0]], device=TEST_DEVICE
        )
        hierarchy = labels([0, 0], [1, 0], [0, 0], [1, 0], [2, 1], [2, 1])
        calculator = AccuracyCalculator(
            include=("mean_average_precision_at_r", "ndcg_at_3"),
            k=3,
            label_comparison_fn=same_class,
            custom_metrics={
                "ndcg_at_3": RetrievalMetric(
                    partial(ndcg, relevance_fn=graded, k=3), neighbors=3
                ),
            },
        )
        binary_only = AccuracyCalculator(
            include=("mean_average_precision_at_r",),
            k=3,
            label_comparison_fn=same_class,
        )
        both = calculator.get_accuracy(embeddings, hierarchy)
        alone = binary_only.get_accuracy(embeddings, hierarchy)
        self.assertEqual(
            both["mean_average_precision_at_r"], alone["mean_average_precision_at_r"]
        )
        self.assertGreater(both["ndcg_at_3"], 0)
        self.assertLess(both["ndcg_at_3"], 1)
