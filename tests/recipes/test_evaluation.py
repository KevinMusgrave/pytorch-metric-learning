import math
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from docs.recipes.evaluation import (
    extract_embeddings,
    nearest_centroid_accuracy,
    parent_conditioned_accuracy,
    split_accuracy,
    two_stream_accuracy,
)
from pytorch_metric_learning.evaluation import AccuracyCalculator

from .. import TEST_DEVICE, TEST_DTYPES


def angles(*values):
    radians = torch.deg2rad(torch.tensor(values, dtype=torch.float64))
    return torch.stack([radians.cos(), radians.sin()], dim=1).to(TEST_DEVICE)


def labels(*values):
    return torch.tensor(values, device=TEST_DEVICE)


class TestEvaluationRecipes(unittest.TestCase):
    def setUp(self):
        self.calculator = AccuracyCalculator(include=("precision_at_1",))

    def test_extraction_preserves_order_and_caller_mode(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5)).to(TEST_DEVICE)
        model.eval()
        data = torch.randn(7, 3, device=TEST_DEVICE)
        targets = torch.arange(7, device=TEST_DEVICE)
        loader = DataLoader(TensorDataset(data, targets), batch_size=3)
        embeddings, extracted = extract_embeddings(model, loader)
        self.assertFalse(model.training)
        self.assertFalse(embeddings.requires_grad)
        self.assertTrue(torch.equal(extracted, targets))
        expected = nn.functional.normalize(model(data), dim=1)
        torch.testing.assert_close(embeddings, expected)

    def test_split_composition(self):
        train = (angles(0, 10, 20, 30, 50, 60, 70, 80), labels(0, 0, 0, 0, 1, 1, 1, 1))
        val = (angles(1, 11, 21, 31, 51, 59, 71, 81), labels(1, 1, 1, 1, 1, 0, 0, 0))
        cases = [
            (train, [], True, 1),
            (val, [], True, 6 / 8),
            (train, [val], True, 1 / 8),
            (val, [train], True, 1 / 8),
            (val, [train], False, 1 / 8),
        ]
        for query, references, query_in_reference, expected in cases:
            with self.subTest(expected=expected, query_in=query_in_reference):
                result = split_accuracy(
                    self.calculator, query, references, query_in_reference
                )
                self.assertAlmostEqual(result["precision_at_1"], expected)

    def test_nearest_centroid_uses_class_means(self):
        reference = torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [4.0, 0.0], [6.0, 0.0], [7.0, 0.0]],
            device=TEST_DEVICE,
        )
        reference_labels = labels(0, 0, 1, 1, 1)
        query = torch.tensor([[5.5, 0.0], [5.0, 0.0]], device=TEST_DEVICE)
        result = nearest_centroid_accuracy(
            self.calculator, (query, labels(1, 0)), (reference, reference_labels)
        )
        self.assertAlmostEqual(result["precision_at_1"], 1.0)
        nearest_example = self.calculator.get_accuracy(
            query, labels(1, 0), reference, reference_labels
        )
        self.assertAlmostEqual(nearest_example["precision_at_1"], 0.5)

    def test_nearest_centroid_matches_direct_computation(self):
        generator = torch.Generator().manual_seed(5)
        reference = torch.randn(60, 4, generator=generator).to(TEST_DEVICE)
        reference_labels = (torch.arange(60) % 7).to(TEST_DEVICE)
        query = torch.randn(40, 4, generator=generator).to(TEST_DEVICE)
        query_labels = (torch.arange(40) % 7).to(TEST_DEVICE)
        centroids = torch.stack(
            [reference[reference_labels == c].mean(dim=0) for c in range(7)]
        )
        nearest = torch.cdist(query, centroids).argmin(dim=1)
        result = nearest_centroid_accuracy(
            self.calculator, (query, query_labels), (reference, reference_labels)
        )
        self.assertAlmostEqual(
            result["precision_at_1"], (nearest == query_labels).float().mean().item()
        )

    def test_nearest_centroid_keeps_sparse_labels_and_excludes_lone_queries(self):
        reference = torch.tensor([[0.0, 0.0], [9.0, 0.0]], device=TEST_DEVICE)
        query = torch.tensor([[1.0, 0.0], [8.0, 0.0], [0.0, 0.0]], device=TEST_DEVICE)
        result = nearest_centroid_accuracy(
            self.calculator,
            (query, labels(7, 2**40, 5)),
            (reference, labels(7, 2**40)),
        )
        self.assertEqual(result["precision_at_1"], 1.0)

    def test_nearest_centroid_keeps_embedding_dtype(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                reference = torch.tensor([[0.0], [2.0]], device=TEST_DEVICE).to(dtype)
                result = nearest_centroid_accuracy(
                    self.calculator,
                    (reference[:1], labels(0)),
                    (reference, labels(0, 0)),
                )
                self.assertEqual(result["precision_at_1"], 1.0)

    def test_two_stream_does_not_infer_self_matches(self):
        anchors = angles(0, 10, 20, 30, 50, 60, 70, 80)
        references = angles(81, 71, 61, 31, 51, 21, 11, 1)
        result = two_stream_accuracy(
            self.calculator, anchors, references, labels(1, 2, 3, 4, 5, 6, 7, 8)
        )
        self.assertAlmostEqual(result["precision_at_1"], 0.25)

    def test_parent_conditioned_metrics(self):
        train = angles(0, 9, 21, 29, 31, 39, 51, 59)
        train_labels = torch.stack(
            [labels(2, 2, 3, 3, 4, 4, 5, 5), labels(0, 0, 0, 0, 1, 1, 1, 1)], dim=1
        )
        val = angles(2, 11, 23, 32, 33, 41, 53, 89, 90)
        val_labels = torch.stack(
            [labels(2, 2, 4, 3, 4, 4, 4, 5, 5), labels(0, 0, 0, 0, 0, 1, 1, 1, 1)],
            dim=1,
        )
        cases = [
            (train, train_labels, train, train_labels, True, 1),
            (val, val_labels, val, val_labels, True, (1 + 0.5) / 2),
            (
                train,
                train_labels,
                torch.cat([train, val]),
                torch.cat([train_labels, val_labels]),
                True,
                (0.75 + 0.5) / 2,
            ),
            (
                val,
                val_labels,
                torch.cat([val, train]),
                torch.cat([val_labels, train_labels]),
                True,
                (0.4 + 0.75) / 2,
            ),
            (val, val_labels, train, train_labels, False, (1 + 0.75) / 2),
        ]
        for query, q_labels, reference, r_labels, query_in, expected in cases:
            with self.subTest(expected=expected):
                result = parent_conditioned_accuracy(
                    self.calculator, query, q_labels, reference, r_labels, query_in
                )
                self.assertEqual(set(result), {"precision_at_1_level0"})
                self.assertTrue(math.isclose(result["precision_at_1_level0"], expected))

    def test_parent_conditioned_levels_are_averaged(self):
        embeddings = angles(0, 1, 40, 41, 90, 91, 130, 131)
        hierarchy = torch.stack(
            [
                labels(0, 0, 1, 1, 2, 2, 3, 3),
                labels(0, 0, 0, 0, 1, 1, 1, 1),
                labels(0, 0, 0, 0, 0, 0, 0, 0),
            ],
            dim=1,
        )
        result = parent_conditioned_accuracy(
            self.calculator, embeddings, hierarchy, embeddings, hierarchy, True
        )
        self.assertEqual(result["precision_at_1_level0"], 1)
        self.assertEqual(result["precision_at_1_level1"], 1)
        self.assertEqual(result["AVERAGE_precision_at_1"], 1)


if __name__ == "__main__":
    unittest.main()
