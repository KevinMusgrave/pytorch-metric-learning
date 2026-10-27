import unittest

import numpy as np
import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils.common_functions import LOGGER
from pytorch_metric_learning.utils.retrieval import CustomKNN

from .. import TEST_DEVICE


class TestCalculateAccuraciesValidK(unittest.TestCase):
    def test_valid_k(self):
        for k in [-1, 0, 1.5, "max"]:
            self.assertRaises(ValueError, accuracy_calculator.AccuracyCalculator, k=k)

    def test_k_warning(self):
        for k in [10, 100, 2000, "max_bin_count"]:
            AC = accuracy_calculator.AccuracyCalculator(k=k)
            embeddings = torch.randn(10000, 32)
            labels = torch.randint(0, 10, size=(10000,))
            if k in [10, 100]:
                with self.assertLogs(LOGGER, level="WARNING"):
                    AC.get_accuracy(embeddings, labels)
            else:
                with self.assertNoLogs(LOGGER, level="WARNING"):
                    AC.get_accuracy(embeddings, labels)


class TestAccuracyCalculatorContracts(unittest.TestCase):
    def test_default_metrics_and_backend(self):
        AC = accuracy_calculator.AccuracyCalculator()
        self.assertEqual(
            set(AC.get_curr_metrics()),
            {
                "mean_average_precision",
                "mean_average_precision_at_r",
                "mean_reciprocal_rank",
                "precision_at_1",
                "r_precision",
            },
        )
        self.assertIsInstance(AC.knn_func, CustomKNN)
        self.assertIsNone(AC.kmeans_func)

    def test_each_metric_can_be_computed_alone(self):
        values = [0.0, 0.7, 1.9, 5.0, 5.6, 1.2]
        label_values = [0, 0, 0, 1, 1, 1]
        expected = dict.fromkeys(self.metric_names(), 0.0)
        for q, (value, label) in enumerate(zip(values, label_values, strict=True)):
            order = sorted(
                (abs(value - other), r) for r, other in enumerate(values) if r != q
            )
            relevant = [label_values[r] == label for _, r in order]
            total = sum(relevant)
            hits = [sum(relevant[: i + 1]) for i in range(len(relevant))]
            precisions = [h / (i + 1) for i, h in enumerate(hits)]
            expected["precision_at_1"] += relevant[0]
            expected["r_precision"] += hits[total - 1] / total
            expected["mean_reciprocal_rank"] += 1 / (relevant.index(True) + 1)
            expected["mean_average_precision"] += (
                sum(p for p, rel in zip(precisions, relevant, strict=True) if rel)
                / total
            )
            expected["mean_average_precision_at_r"] += (
                sum(
                    p
                    for p, rel in zip(precisions[:total], relevant[:total], strict=True)
                    if rel
                )
                / total
            )
        embeddings = torch.tensor(values).unsqueeze(1)
        labels = torch.tensor(label_values)
        for metric in expected:
            with self.subTest(metric=metric):
                calculator = accuracy_calculator.AccuracyCalculator(include=(metric,))
                result = calculator.get_accuracy(embeddings, labels)
                self.assertEqual(set(result), {metric})
                self.assertAlmostEqual(result[metric], expected[metric] / len(values))

    @staticmethod
    def metric_names():
        return (
            "mean_average_precision",
            "mean_average_precision_at_r",
            "mean_reciprocal_rank",
            "precision_at_1",
            "r_precision",
        )

    def test_clustering_requires_explicit_kmeans(self):
        with self.assertRaisesRegex(ValueError, "kmeans_func"):
            accuracy_calculator.AccuracyCalculator(include=("NMI",))
        accuracy_calculator.AccuracyCalculator(
            include=("NMI",), kmeans_func=lambda x, n: torch.zeros(len(x))
        )

    def test_device_argument_and_input_device(self):
        embeddings = torch.tensor([[0.0], [0.1], [5.0], [5.1]], device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        seen = []

        def knn_func(query, k, reference, ref_includes_query):
            seen.append(query.device)
            return CustomKNN(LpDistance(normalize_embeddings=False))(
                query, k, reference, ref_includes_query
            )

        for device in (None, str(TEST_DEVICE), TEST_DEVICE):
            AC = accuracy_calculator.AccuracyCalculator(
                include=("precision_at_1",), device=device, knn_func=knn_func
            )
            accuracies = AC.get_accuracy(embeddings, labels)
            self.assertEqual(accuracies["precision_at_1"], 1)
        self.assertEqual(seen, [embeddings.device] * 3)

    def test_non_finite_embeddings_are_rejected(self):
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",))
        labels = torch.tensor([0, 0, 1])
        for value in (float("nan"), float("inf")):
            embeddings = torch.tensor([[0.0], [value], [1.0]])
            with self.assertRaisesRegex(ValueError, "finite"):
                AC.get_accuracy(embeddings, labels)


class TestEmbeddingsComeFromSameSource(unittest.TestCase):
    def test_tied_distances(self):
        colocated_emb_with_different_labels = torch.tensor([[10], [10], [20], [20]])
        labels = torch.tensor([1, 0, 1, 0])
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",))
        accuracies = AC.get_accuracy(colocated_emb_with_different_labels, labels)
        self.assertEqual(accuracies["precision_at_1"], 0)

    def test_many_tied_distances(self):
        emb = torch.zeros(10000, 2)
        labels = torch.randint(0, 2, size=(10000,))
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",), k=1)
        accuracies = AC.get_accuracy(emb, labels)
        self.assertTrue(np.isclose(accuracies["precision_at_1"], 0.5, rtol=0.1))

    def test_query_within_reference(self):
        query = torch.tensor([0, 10, 20]).unsqueeze(1)
        ref = torch.tensor([0, 5, 10, 15, 20]).unsqueeze(1)
        query_labels = torch.tensor([0, 1, 2])
        ref_labels = torch.tensor([0, 9, 1, 8, 2])
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",))

        AC.get_accuracy(query, query_labels, ref, ref_labels, ref_includes_query=False)

        with self.assertRaises(ValueError):
            AC.get_accuracy(
                query, query_labels, ref, ref_labels, ref_includes_query=True
            )

        ref_starting_with_query = torch.tensor([0, 10, 20, 5, 15]).unsqueeze(1)
        ref_labels[:3] = query_labels
        AC.get_accuracy(
            query,
            query_labels,
            ref_starting_with_query,
            ref_labels,
            ref_includes_query=True,
        )
