import logging
import unittest
from typing import Any

import numpy as np
import pytest
import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.evaluation import accuracy_calculator
from pytorch_metric_learning.evaluation.retrieval import CustomKNN, FaissKNN

from .. import TEST_DEVICE


class TestMetricDevice(unittest.TestCase):
    def test_float64_devices_keep_metric_work_on_the_input_device(self):
        self.assertEqual(
            accuracy_calculator._metric_device(torch.device("cuda", 1)),
            torch.device("cuda", 1),
        )
        self.assertEqual(
            accuracy_calculator._metric_device(torch.device("mps")),
            torch.device("cpu"),
        )

    @pytest.mark.mps
    def test_metrics_run_with_mps_inputs(self):
        embeddings = torch.tensor([[0.0], [0.1], [5.0], [5.1], [9.0]])
        labels = torch.tensor([0, 0, 1, 1, 1])
        calculator = accuracy_calculator.AccuracyCalculator()
        expected = calculator.get_accuracy(embeddings, labels)
        result = calculator.get_accuracy(embeddings.to("mps"), labels.to("mps"))
        self.assertEqual(result, expected)
        per_class = calculator.get_accuracy_per_class(
            embeddings.to("mps"), labels.to("mps")
        )
        self.assertEqual(
            per_class, calculator.get_accuracy_per_class(embeddings, labels)
        )


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
                with self.assertLogs("pytorch_metric_learning", level="WARNING"):
                    AC.get_accuracy(embeddings, labels)
            else:
                with self.assertNoLogs("pytorch_metric_learning", level="WARNING"):
                    AC.get_accuracy(embeddings, labels)


class TestAccuracyCalculatorContracts(unittest.TestCase):
    def test_single_embedding_has_no_neighbors_for_every_k(self):
        embeddings = torch.randn(1, 4)
        labels = torch.tensor([0])
        for k in (None, "max_bin_count", 1):
            with self.subTest(k=k):
                AC = accuracy_calculator.AccuracyCalculator(k=k)
                accuracies = AC.get_accuracy(embeddings, labels)
                self.assertTrue(all(np.isnan(v) for v in accuracies.values()))

    def test_default_metrics_and_backend(self):
        AC = accuracy_calculator.AccuracyCalculator()
        self.assertEqual(
            set(AC.metrics),
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


def brute_force_retrieval(positions, labels, comparison):
    r_precisions, maps_at_r, maps = [], [], []
    for query in range(len(positions)):
        others = [i for i in range(len(positions)) if i != query]
        others.sort(key=lambda i: (abs(positions[i] - positions[query]), i))
        hits = [bool(comparison(labels[query], labels[i])) for i in others]
        relevant = sum(hits)
        if relevant == 0:
            continue
        precisions = [sum(hits[: k + 1]) / (k + 1) for k in range(len(hits))]
        r_precisions.append(sum(hits[:relevant]) / relevant)
        maps_at_r.append(
            sum(p for p, h in zip(precisions[:relevant], hits, strict=False) if h)
            / relevant
        )
        maps.append(
            sum(p for p, h in zip(precisions, hits, strict=True) if h) / relevant
        )
    return (np.mean(r_precisions), np.mean(maps_at_r), np.mean(maps))


class TestComparatorAndPerClassResults(unittest.TestCase):
    def test_custom_comparator_that_rejects_self_matches(self):
        def comparison(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

        positions = [0.0, 0.1, 0.25, 0.5, 3.0, 3.2, 3.3, 9.0]
        labels = torch.tensor(
            [[0, 0], [0, 1], [0, 0], [1, 2], [1, 3], [1, 2], [2, 0], [2, 0]],
            device=TEST_DEVICE,
        )
        embeddings = torch.tensor(positions, device=TEST_DEVICE).unsqueeze(1)
        calculator = accuracy_calculator.AccuracyCalculator(
            include=(
                "r_precision",
                "mean_average_precision_at_r",
                "mean_average_precision",
            ),
            label_comparison_fn=comparison,
            device=TEST_DEVICE,
        )
        result = calculator.get_accuracy(embeddings, labels)
        expected = brute_force_retrieval(positions, labels.tolist(), comparison_python)
        for name, value in zip(
            ("r_precision", "mean_average_precision_at_r", "mean_average_precision"),
            expected,
            strict=True,
        ):
            self.assertTrue(np.isfinite(result[name]), name)
            self.assertAlmostEqual(result[name], value, msg=name)

    def test_per_class_results_are_keyed_by_label(self):
        calculator = accuracy_calculator.AccuracyCalculator(
            include=("precision_at_1",), device=TEST_DEVICE
        )
        query = torch.tensor([[0.0], [5.0], [5.1]], device=TEST_DEVICE)
        reference = torch.tensor([[0.1], [10.0]], device=TEST_DEVICE)
        result = calculator.get_accuracy_per_class(
            query,
            torch.tensor([3, 7, 7], device=TEST_DEVICE),
            reference,
            torch.tensor([3, 9], device=TEST_DEVICE),
        )
        values = result["precision_at_1"]
        self.assertEqual(list(values), [3, 7])
        self.assertEqual(values[3], 1.0)
        self.assertTrue(np.isnan(values[7]))
        none = calculator.get_accuracy_per_class(
            query,
            torch.tensor([3, 7, 7], device=TEST_DEVICE),
            reference,
            torch.tensor([1, 2], device=TEST_DEVICE),
        )["precision_at_1"]
        self.assertEqual(list(none), [3, 7])
        self.assertTrue(all(np.isnan(v) for v in none.values()))

    def test_repeated_calls_do_not_change_selected_metrics(self):
        calculator = accuracy_calculator.AccuracyCalculator(device=TEST_DEVICE)
        embeddings = torch.randn(10, 3, device=TEST_DEVICE)
        labels = torch.randint(0, 3, (10,), device=TEST_DEVICE)
        first = calculator.get_accuracy(embeddings, labels, include=("precision_at_1",))
        second = calculator.get_accuracy(embeddings, labels)
        self.assertEqual(list(first), ["precision_at_1"])
        self.assertEqual(set(second), set(calculator.metrics))


def comparison_python(x, y):
    return x[0] == y[0] and x[1] != y[1]


def ranked_relevance(positions, labels, query, k):
    others = [i for i in range(len(positions)) if i != query]
    others.sort(key=lambda i: (abs(positions[i] - positions[query]), i))
    hits = [labels[i] == labels[query] for i in others]
    return hits[: len(hits) if k is None else k], sum(hits)


def reference_metric(name, hits, relevant):
    precisions = [sum(hits[: i + 1]) / (i + 1) for i in range(len(hits))]
    within_r = hits[:relevant]
    if name == "precision_at_1":
        return float(hits[0])
    if name == "precision_at_2":
        return sum(hits[:2]) / 2
    if name == "r_precision":
        return sum(within_r) / relevant
    if name == "mean_average_precision_at_r":
        return (
            sum(p for p, h in zip(precisions, within_r, strict=False) if h) / relevant
        )
    if name == "mean_average_precision":
        return sum(p for p, h in zip(precisions, hits, strict=True) if h) / relevant
    return next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)


def reference_results(positions, labels, names, k):
    per_class = {name: {} for name in names}
    for query in range(len(positions)):
        hits, relevant = ranked_relevance(positions, labels, query, k)
        for name in names:
            values = per_class[name].setdefault(labels[query], [])
            if relevant:
                values.append(reference_metric(name, hits, relevant))
    means = {
        name: {
            label: float(np.mean(v)) if v else float("nan")
            for label, v in sorted(values.items())
        }
        for name, values in per_class.items()
    }
    pooled = {
        name: float(np.mean([x for v in values.values() for x in v]))
        for name, values in per_class.items()
    }
    avg_of_avgs = {
        name: float(np.nanmean(list(values.values()))) for name, values in means.items()
    }
    return pooled, avg_of_avgs, means


def precision_at_2(inputs):
    return inputs.matches[:, :2].sum(dim=1, dtype=torch.float64) / 2


class TestMetricExtensionContract(unittest.TestCase):
    names = (
        "mean_average_precision",
        "mean_average_precision_at_r",
        "mean_reciprocal_rank",
        "precision_at_1",
        "precision_at_2",
        "r_precision",
    )

    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.positions = torch.rand(40, generator=generator, dtype=torch.float64)
        self.labels = torch.randint(0, 5, (40,), generator=generator)
        self.labels[-1] = 9
        self.embeddings = self.positions.unsqueeze(1).to(TEST_DEVICE)
        self.custom = {
            "precision_at_2": accuracy_calculator.RetrievalMetric(
                precision_at_2, neighbors=2
            )
        }

    def test_values_match_float64_reference_for_every_reduction(self):
        logger = logging.getLogger("pytorch_metric_learning")
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel(logging.ERROR)
        for k in (None, 3):
            expected = reference_results(
                self.positions.tolist(), self.labels.tolist(), self.names, k
            )
            for avg_of_avgs in (False, True):
                with self.subTest(k=k, avg_of_avgs=avg_of_avgs):
                    calculator = accuracy_calculator.AccuracyCalculator(
                        k=k, avg_of_avgs=avg_of_avgs, custom_metrics=self.custom
                    )
                    self.assertEqual(set(calculator.metrics), set(self.names))
                    labels = self.labels.to(TEST_DEVICE)
                    result = calculator.get_accuracy(self.embeddings, labels)
                    reference = expected[int(avg_of_avgs)]
                    for name in self.names:
                        self.assertAlmostEqual(result[name], reference[name], 12)
                    per_class = calculator.get_accuracy_per_class(
                        self.embeddings, labels
                    )
                    for name in self.names:
                        self.assertEqual(list(per_class[name]), [0, 1, 2, 3, 4, 9])
                        np.testing.assert_allclose(
                            list(per_class[name].values()),
                            list(expected[2][name].values()),
                            rtol=1e-12,
                        )

    def test_neighbors_retrieved_follow_selected_metrics(self):
        labels = self.labels.to(TEST_DEVICE)
        max_relevant = int(torch.bincount(self.labels).max()) - 1
        available = len(self.labels) - 1
        cases = (
            (("precision_at_1",), None, 1),
            (("precision_at_2",), None, 2),
            (("precision_at_1", "precision_at_2"), 1, 1),
            (("r_precision",), None, max_relevant),
            (("mean_average_precision_at_r",), "max_bin_count", max_relevant),
            (("mean_average_precision",), None, available),
            (("mean_reciprocal_rank",), 5, 5),
            (("mean_average_precision",), "max_bin_count", max_relevant),
            (("precision_at_1", "r_precision"), None, max_relevant),
        )
        for include, k, expected in cases:
            with self.subTest(include=include, k=k):
                requested = []

                def knn_func(
                    query, num_k, reference, ref_includes_query, requested=requested
                ):
                    requested.append(num_k)
                    return CustomKNN(LpDistance(normalize_embeddings=False))(
                        query, num_k, reference, ref_includes_query
                    )

                calculator = accuracy_calculator.AccuracyCalculator(
                    include=include,
                    k=k,
                    knn_func=knn_func,
                    custom_metrics=self.custom,
                )
                calculator.get_accuracy(self.embeddings, labels)
                calculator.get_accuracy_per_class(self.embeddings, labels)
                self.assertEqual(requested, [expected, expected])

    def test_undefined_values_are_left_out_of_every_reduction(self):
        def first_class_only(inputs):
            values = inputs.matches[:, 0].to(torch.float64)
            return torch.where(inputs.query_labels == 0, values, torch.nan)

        calculator = accuracy_calculator.AccuracyCalculator(
            include=("first_class_only",),
            custom_metrics={
                "first_class_only": accuracy_calculator.RetrievalMetric(
                    first_class_only, neighbors=1
                )
            },
        )
        embeddings = torch.tensor(
            [[0.0], [0.1], [0.3], [5.0], [5.1]], device=TEST_DEVICE
        )
        labels = torch.tensor([0, 0, 1, 1, 1], device=TEST_DEVICE)
        self.assertEqual(
            calculator.get_accuracy(embeddings, labels), {"first_class_only": 1.0}
        )
        per_class = calculator.get_accuracy_per_class(embeddings, labels)
        self.assertEqual(per_class["first_class_only"][0], 1.0)
        self.assertTrue(np.isnan(per_class["first_class_only"][1]))

    def test_clustering_metrics_are_registered_without_per_class_values(self):
        def purity(query_labels, cluster_labels):
            total = 0
            for cluster in torch.unique(cluster_labels):
                members = query_labels[cluster_labels == cluster]
                total += int(torch.bincount(members).max())
            return total / len(query_labels)

        calculator = accuracy_calculator.AccuracyCalculator(
            include=("precision_at_1", "purity"),
            kmeans_func=lambda x, n: (x[:, 0] > 0.05).long(),
            custom_metrics={"purity": accuracy_calculator.ClusteringMetric(purity)},
        )
        embeddings = torch.tensor([[0.0], [0.1], [5.0], [5.1]], device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        self.assertEqual(
            calculator.get_accuracy(embeddings, labels),
            {"precision_at_1": 1.0, "purity": 0.75},
        )
        self.assertEqual(
            calculator.get_accuracy_per_class(embeddings, labels),
            {"precision_at_1": {0: 1.0, 1: 1.0}},
        )
        with self.assertRaisesRegex(ValueError, "per-class"):
            calculator.get_accuracy_per_class(embeddings, labels, include=("purity",))
        with self.assertRaisesRegex(ValueError, "kmeans_func"):
            accuracy_calculator.AccuracyCalculator(
                include=("purity",),
                custom_metrics={"purity": accuracy_calculator.ClusteringMetric(purity)},
            )
        default = accuracy_calculator.AccuracyCalculator(
            custom_metrics={"purity": accuracy_calculator.ClusteringMetric(purity)}
        )
        self.assertNotIn("purity", default.metrics)

    def test_invalid_registrations_and_outputs_are_rejected(self):
        invalid: tuple[Any, ...] = (0, -1, True, 1.5, "max")
        for neighbors in invalid:
            with (
                self.subTest(neighbors=neighbors),
                self.assertRaisesRegex(ValueError, "neighbors"),
            ):
                accuracy_calculator.RetrievalMetric(precision_at_2, neighbors)
        with self.assertRaisesRegex(ValueError, "built-in"):
            accuracy_calculator.AccuracyCalculator(
                custom_metrics={"precision_at_1": self.custom["precision_at_2"]}
            )
        unregistered: Any = {"precision_at_2": precision_at_2}
        with self.assertRaises(TypeError):
            accuracy_calculator.AccuracyCalculator(custom_metrics=unregistered)
        outputs: tuple[Any, ...] = (
            lambda inputs: 1.0,
            lambda inputs: inputs.matches.double(),
        )
        for output in outputs:
            calculator = accuracy_calculator.AccuracyCalculator(
                include=("bad",),
                custom_metrics={"bad": accuracy_calculator.RetrievalMetric(output)},
            )
            with self.assertRaisesRegex(ValueError, "one value per query"):
                calculator.get_accuracy(self.embeddings, self.labels.to(TEST_DEVICE))


class TestEvaluationBoundary(unittest.TestCase):
    def test_metrics_and_search_build_no_autograd_graph(self):
        embeddings = torch.tensor(
            [[0.0], [0.1], [5.0], [5.1]], device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        grad_enabled = []

        def knn_func(query, k, reference, ref_includes_query):
            grad_enabled.append(torch.is_grad_enabled())
            return CustomKNN(LpDistance(normalize_embeddings=False))(
                query, k, reference, ref_includes_query
            )

        def kmeans_func(x, num_clusters):
            grad_enabled.append(torch.is_grad_enabled())
            return (x[:, 0] > 2.5).long()

        calculator = accuracy_calculator.AccuracyCalculator(
            include=("precision_at_1", "NMI"),
            knn_func=knn_func,
            kmeans_func=kmeans_func,
        )
        calculator.get_accuracy(embeddings, labels, include=("precision_at_1",))
        calculator.get_accuracy_per_class(embeddings, labels)
        self.assertEqual(grad_enabled, [False, False])
        for batch_size in (None, 2):
            distances, _ = CustomKNN(LpDistance(), batch_size)(
                embeddings, 2, embeddings
            )
            self.assertFalse(distances.requires_grad)
            self.assertIsNone(distances.grad_fn)

    def test_neighbor_indices_outside_reference_raise_value_error(self):
        reference = torch.tensor([[0.0], [1.0], [2.0]], device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        for shift in (-1, 3):

            def knn_func(query, k, reference, ref_includes_query, shift=shift):
                distances, indices = CustomKNN(LpDistance())(query, k, reference)
                return distances, indices + shift

            calculator = accuracy_calculator.AccuracyCalculator(knn_func=knn_func)
            with (
                self.subTest(shift=shift),
                self.assertRaisesRegex(ValueError, "3 rows"),
            ):
                calculator.get_accuracy(reference, labels, reference, labels)

    def test_neighbor_indices_of_another_shape_raise_value_error(self):
        reference = torch.tensor([[0.0], [0.1], [5.0], [5.1]], device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)

        def knn_func(query, k, reference, ref_includes_query):
            distances, indices = CustomKNN(LpDistance())(
                query, k, reference, ref_includes_query
            )
            return distances[:, :-1], indices[:, :-1]

        calculator = accuracy_calculator.AccuracyCalculator(
            include=("r_precision",), knn_func=knn_func
        )
        with self.assertRaisesRegex(ValueError, r"require shape \(4, 1\)"):
            calculator.get_accuracy(reference, labels)

    def test_boolean_labels_match_integer_labels(self):
        embeddings = torch.tensor([[0.0], [0.1], [5.0], [5.1]], device=TEST_DEVICE)
        labels = torch.tensor([False, False, True, True], device=TEST_DEVICE)
        calculator = accuracy_calculator.AccuracyCalculator(
            include=("precision_at_1", "r_precision", "mean_average_precision")
        )
        self.assertEqual(
            calculator.get_accuracy(embeddings, labels),
            calculator.get_accuracy(embeddings, labels.long()),
        )
        self.assertEqual(
            calculator.get_accuracy(embeddings, labels)["precision_at_1"], 1
        )

    def test_persistent_faiss_index_is_reported_instead_of_index_error(self):
        reference = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        labels = torch.tensor([0, 0, 1, 1])
        knn = FaissKNN(reset_before=False, reset_after=False)
        knn.train(reference + 100)
        calculator = accuracy_calculator.AccuracyCalculator(
            include=("precision_at_1",), knn_func=knn
        )
        with self.assertRaisesRegex(ValueError, "reset_before=False"):
            calculator.get_accuracy(reference, labels, reference, labels)


class TestRelevantCounts(unittest.TestCase):
    def test_default_counts_match_float64_reference(self):
        generator = torch.Generator().manual_seed(5)
        reference = torch.randint(0, 6, (50,), generator=generator)
        query = torch.tensor([0, 1, 2, 3, 4, 5, 7, -2])
        for query_dtype, reference_dtype in (
            (torch.int64, torch.int64),
            (torch.int64, torch.int32),
            (torch.float64, torch.int64),
            (torch.int32, torch.float32),
        ):
            for ref_includes_query in (False, True):
                with self.subTest(
                    dtypes=(query_dtype, reference_dtype),
                    self_search=ref_includes_query,
                ):
                    counts = accuracy_calculator._relevant_counts(
                        query.to(query_dtype).to(TEST_DEVICE),
                        reference.to(reference_dtype).to(TEST_DEVICE),
                        ref_includes_query,
                        None,
                    )
                    expected = [
                        float(sum(float(r) == float(q) for r in reference.tolist()))
                        - ref_includes_query
                        for q in query.tolist()
                    ]
                    self.assertEqual(counts.dtype, torch.long)
                    self.assertEqual(counts.cpu().double().tolist(), expected)
        empty = accuracy_calculator._relevant_counts(
            query.to(TEST_DEVICE), query[:0].to(TEST_DEVICE), False, None
        )
        self.assertEqual(empty.tolist(), [0] * len(query))
