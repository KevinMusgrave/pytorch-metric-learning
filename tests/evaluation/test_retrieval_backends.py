import gc
import importlib
import importlib.util
import math
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.evaluation import accuracy_calculator, retrieval

from .. import TEST_DEVICE, TEST_DTYPES

HAS_FAISS = importlib.util.find_spec("faiss") is not None


class TestRetrievalBackends(unittest.TestCase):
    def test_fixed_k_adapts_to_each_gallery_and_self_exclusion(self):
        embeddings = torch.tensor([[0.0], [2.0], [3.0], [7.0]], device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        query = embeddings[:1]
        query_labels = labels[2:3]
        reference = torch.tensor([[0.1], [1.0], [3.0]], device=TEST_DEVICE)
        reference_labels = labels[:3]
        for k in (1, 2, 3, 4, 1000):
            calculator = accuracy_calculator.AccuracyCalculator(k=k)
            cases = (
                (
                    "one reference",
                    (query, query_labels, reference[-1:], reference_labels[-1:]),
                    1.0,
                    1.0,
                ),
                (
                    "three references",
                    (query, query_labels, reference, reference_labels),
                    0.0,
                    1 / 3 if k >= 3 else 0.0,
                ),
                (
                    "self search",
                    (embeddings, labels),
                    0.5,
                    {1: 0.5, 2: 0.625}.get(k, 17 / 24),
                ),
                (
                    "query prefix",
                    (embeddings[:2], labels[:2], embeddings, labels, True),
                    0.5,
                    0.75 if k >= 2 else 0.5,
                ),
            )
            for name, args, precision, rank_score in cases:
                with self.subTest(k=k, gallery=name):
                    result = calculator.get_accuracy(*args)
                    for metric in (
                        "precision_at_1",
                        "r_precision",
                        "mean_average_precision_at_r",
                    ):
                        self.assertAlmostEqual(result[metric], precision)
                    for metric in ("mean_average_precision", "mean_reciprocal_rank"):
                        self.assertAlmostEqual(result[metric], rank_score)

    def test_fixed_k_singleton_self_search_has_no_valid_neighbors(self):
        embeddings = torch.tensor([[0.0]], device=TEST_DEVICE)
        labels = torch.tensor([0], device=TEST_DEVICE)
        for k in (None, 1, 4):
            with self.subTest(k=k):
                result = accuracy_calculator.AccuracyCalculator(k=k).get_accuracy(
                    embeddings, labels
                )
                self.assertEqual(len(result), 5)
                self.assertTrue(all(math.isnan(value) for value in result.values()))

    def test_calculator_releases_knn_outputs_after_each_call(self):
        query = torch.arange(257, device=TEST_DEVICE, dtype=torch.float32)[:, None]
        labels = torch.arange(len(query), device=TEST_DEVICE) // 2
        for k in (None, 3):
            with self.subTest(k=k):
                calculator = accuracy_calculator.AccuracyCalculator(
                    include=("precision_at_1",), k=k
                )
                search = calculator.knn_func
                outputs = []

                def track_outputs(*args, search=search, outputs=outputs):
                    result = search(*args)
                    outputs.extend(weakref.ref(tensor) for tensor in result)
                    return result

                calculator.knn_func = track_outputs
                for shift in (0.1, 0.2):
                    with torch.no_grad():
                        metrics = calculator.get_accuracy(
                            query, labels, query + shift, labels
                        )
                    self.assertEqual(metrics["precision_at_1"], 1.0)
                    self.assertTrue(outputs)
                    gc.collect()
                    self.assertTrue(all(output() is None for output in outputs))

    def test_failed_batched_search_releases_inputs_and_recovers(self):
        class CompleteBatchDistance(LpDistance):
            def forward(self, query_emb, ref_emb=None):
                if len(query_emb) != 2:
                    raise ValueError("expected a complete batch")
                return super().forward(query_emb, ref_emb)

        knn = retrieval.CustomKNN(
            CompleteBatchDistance(normalize_embeddings=False), batch_size=2
        )
        query = torch.tensor(
            [[0.0], [1.0], [2.0]], device=TEST_DEVICE, requires_grad=True
        )
        reference = torch.tensor([[0.1], [1.1], [2.1], [3.1]], device=TEST_DEVICE)
        query_reference = weakref.ref(query)
        with self.assertRaisesRegex(ValueError, "complete batch"):
            knn(query, 1, reference)
        del query
        gc.collect()
        self.assertIsNone(query_reference())
        distances, indices = knn(reference - 0.1, 1, reference)
        torch.testing.assert_close(distances, torch.full_like(distances, 0.1))
        self.assertEqual(indices[:, 0].tolist(), [0, 1, 2, 3])

    def test_custom_knn_matches_exhaustive_reference(self):
        torch.manual_seed(11)
        for dtype in TEST_DTYPES:
            query = torch.randn(7, 3, device=TEST_DEVICE, dtype=dtype)
            reference = torch.randn(9, 3, device=TEST_DEVICE, dtype=dtype)
            expected = torch.cdist(query.double(), reference.double()).square()
            expected_distances, expected_indices = expected.topk(4, largest=False)
            for batch_size in (None, 2, 32):
                with self.subTest(dtype=dtype, batch_size=batch_size):
                    knn = retrieval.CustomKNN(
                        LpDistance(normalize_embeddings=False, power=2), batch_size
                    )
                    distances, indices = knn(query, 4, reference)
                    self.assertEqual(distances.dtype, dtype)
                    self.assertEqual(indices.dtype, torch.long)
                    self.assertTrue(torch.equal(indices, expected_indices))
                    tolerance = 1e-2 if dtype == torch.float16 else 1e-4
                    torch.testing.assert_close(
                        distances.double(),
                        expected_distances,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_custom_knn_removes_exact_self_match(self):
        embeddings = torch.tensor([[0.0], [0.0], [3.0]], device=TEST_DEVICE)
        knn = retrieval.CustomKNN(LpDistance(normalize_embeddings=False))
        _, indices = knn(embeddings, 2, embeddings, ref_includes_query=True)
        self.assertEqual(indices.shape, (3, 2))
        for row, neighbors in enumerate(indices.tolist()):
            self.assertNotIn(row, neighbors)

    def test_batched_cosine_search_orders_by_similarity(self):
        query = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], device=TEST_DEVICE)
        reference = torch.tensor(
            [[3.0, 0.0], [0.0, 2.0], [-4.0, 0.0]], device=TEST_DEVICE
        )
        search = retrieval.CustomKNN(CosineSimilarity(), batch_size=1)
        distances, indices = search(query, 2, reference)
        self.assertEqual(indices.tolist(), [[0, 1], [2, 1]])
        torch.testing.assert_close(
            distances, query.new_tensor([[1.0, 0.0], [1.0, 0.0]])
        )

    def test_match_finder_returns_tensors_on_input_device(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=TEST_DEVICE)
        reference = torch.tensor(
            [[1.0, 0.1], [-1.0, 0.0], [0.1, 1.0]], device=TEST_DEVICE
        )
        finder = retrieval.MatchFinder(CosineSimilarity(), threshold=0.9)
        matches = finder.get_match_matrix(query, reference)
        self.assertEqual(matches.device, query.device)
        self.assertTrue(
            torch.equal(
                matches.cpu(),
                torch.tensor([[True, False, False], [False, False, True]]),
            )
        )
        indices = finder.get_match_indices(query, reference)
        self.assertEqual(indices.device, query.device)
        self.assertEqual(indices.dtype, torch.long)
        self.assertEqual(indices.tolist(), [[0, 0], [1, 2]])
        self.assertEqual(finder.get_match_indices(query).tolist(), [[0, 0], [1, 1]])
        distance_finder = retrieval.MatchFinder(LpDistance(), threshold=0.5)
        self.assertEqual(
            distance_finder.is_match(query, reference[[0, 1]]).tolist(), [True, False]
        )
        with self.assertRaisesRegex(ValueError, "threshold"):
            retrieval.MatchFinder(LpDistance()).get_match_matrix(query)

    def test_row_matching_honors_distance_normalization_and_power(self):
        cases = (
            (CosineSimilarity(), 0.9, [0.1, 0.0], [0.1, 0.0], True),
            (CosineSimilarity(), 0.9, [10.0, 0.0], [1.0, 10.0], False),
            (LpDistance(), 0.5, [1.0, 0.0], [10.0, 0.0], True),
            (
                LpDistance(normalize_embeddings=False, power=2),
                3.0,
                [0.0, 0.0],
                [2.0, 0.0],
                False,
            ),
        )
        for dtype in TEST_DTYPES:
            for distance, threshold, x, y, expected in cases:
                with self.subTest(dtype=dtype, distance=distance, x=x, y=y):
                    query = torch.tensor([x], device=TEST_DEVICE, dtype=dtype)
                    reference = torch.tensor([y], device=TEST_DEVICE, dtype=dtype)
                    finder = retrieval.MatchFinder(distance, threshold=threshold)
                    self.assertEqual(
                        finder.is_match(query, reference).tolist(), [expected]
                    )
                    self.assertEqual(
                        finder.get_match_matrix(query, reference).tolist(),
                        [[expected]],
                    )

    def test_missing_optional_backends_are_explicit(self):
        import_module = importlib.import_module

        def missing(name, *args, **kwargs):
            if name.split(".")[0] in {"faiss", "sklearn"}:
                raise ModuleNotFoundError(name=name)
            return import_module(name, *args, **kwargs)

        with mock.patch("importlib.import_module", side_effect=missing):
            with self.assertRaisesRegex(ModuleNotFoundError, r"\[faiss\]"):
                retrieval.FaissKNN()
            calculator = accuracy_calculator.AccuracyCalculator(
                include=("NMI",), kmeans_func=lambda x, n: torch.zeros(len(x))
            )
            embeddings = torch.randn(4, 2, device=TEST_DEVICE)
            labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
            with self.assertRaisesRegex(ModuleNotFoundError, r"\[clustering\]"):
                calculator.get_accuracy(embeddings, labels)

    @unittest.skipUnless(HAS_FAISS, "requires the faiss extra")
    def test_faiss_index_persistence_and_incremental_add(self):
        torch.manual_seed(13)
        reference = torch.randn(20, 4)
        query = reference[:3] + 1e-3
        knn = retrieval.FaissKNN(reset_before=False, reset_after=False)
        with self.assertRaisesRegex(ValueError, "initialized"):
            knn(query, 2)
        knn.train(reference[:10])
        knn.add(reference[10:])
        _, indices = knn(query, 2)
        self.assertEqual(indices[:, 0].tolist(), [0, 1, 2])
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "reference.index")
            knn.save(filename)
            restored = retrieval.FaissKNN(reset_before=False, reset_after=False)
            restored.load(filename)
            restored_distances, restored_indices = restored(query, 2)
        distances, indices = knn(query, 2)
        self.assertTrue(torch.equal(restored_indices, indices))
        torch.testing.assert_close(restored_distances, distances)


if __name__ == "__main__":
    unittest.main()
