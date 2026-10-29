import itertools
import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE


def reference_triplets(
    query: list[int], reference: list[int], same_set: bool
) -> list[tuple[int, int, int]]:
    return [
        (a, p, n)
        for a, query_label in enumerate(query)
        for p, positive_label in enumerate(reference)
        for n, negative_label in enumerate(reference)
        if query_label == positive_label
        and query_label != negative_label
        and (not same_set or a != p)
    ]


class TestTripletEnumeration(unittest.TestCase):
    def check_indices(
        self, labels: torch.Tensor, reference: torch.Tensor | None
    ) -> None:
        other = labels if reference is None else reference
        expected = torch.tensor(
            reference_triplets(labels.tolist(), other.tolist(), reference is None),
            device=TEST_DEVICE,
            dtype=torch.long,
        ).reshape(-1, 3)
        actual = indexing.get_all_triplets_indices(labels, reference)
        for observed, wanted in zip(actual, expected.unbind(dim=1), strict=True):
            torch.testing.assert_close(observed, wanted)

    def test_self_and_separate_set_identity(self) -> None:
        for size in range(5):
            for values in itertools.product(range(2), repeat=size):
                labels = torch.tensor(values, device=TEST_DEVICE, dtype=torch.long)
                with self.subTest(values=values):
                    self.check_indices(labels, None)
                    self.check_indices(labels, labels)
                    self.check_indices(labels, labels.clone())

    def test_rectangular_and_empty_reference_sets(self) -> None:
        for query_values in itertools.product(range(3), repeat=3):
            labels = torch.tensor(query_values, device=TEST_DEVICE)
            for size in range(3):
                for values in itertools.product(range(3), repeat=size):
                    reference = torch.tensor(
                        values, device=TEST_DEVICE, dtype=torch.long
                    )
                    self.check_indices(labels, reference)
        self.check_indices(
            torch.empty(0, device=TEST_DEVICE), torch.tensor([0, 1], device=TEST_DEVICE)
        )

    def test_label_dtypes_and_noncontiguous_views(self) -> None:
        for dtype in (torch.int8, torch.int64, torch.float32, torch.float64):
            labels = torch.tensor(
                [0, 7, 1, 7, 0, 7, 2, 7], device=TEST_DEVICE, dtype=dtype
            )[::2]
            reference = torch.tensor(
                [0, 9, 2, 9, 1, 9, 1, 9], device=TEST_DEVICE, dtype=dtype
            )[::2]
            self.check_indices(labels, None)
            self.check_indices(labels, reference)

    def test_triplet_loss_and_reference_gradients(self) -> None:
        query_labels = [0, 0, 1, 1]
        for separate in (False, True):
            query = torch.tensor(
                [[0.1, 0.4], [0.9, 0.7], [-0.7, 0.1], [0.0, -0.4]],
                device=TEST_DEVICE,
                dtype=torch.float64,
                requires_grad=True,
            )
            reference_labels = [0, 1, 0] if separate else query_labels
            reference = (
                (query.detach()[:3] + 0.3).requires_grad_() if separate else query
            )
            triplets = reference_triplets(query_labels, reference_labels, not separate)
            violations = torch.stack(
                [
                    torch.linalg.vector_norm(query[a] - reference[p])
                    - torch.linalg.vector_norm(query[a] - reference[n])
                    + 0.4
                    for a, p, n in triplets
                ]
            ).clamp_min(0)
            expected = violations[violations > 0].mean()
            loss = TripletMarginLoss(
                margin=0.4, distance=LpDistance(normalize_embeddings=False)
            )
            labels = torch.tensor(query_labels, device=TEST_DEVICE)
            actual = loss(
                query,
                labels,
                ref_emb=reference if separate else None,
                ref_labels=torch.tensor(reference_labels, device=TEST_DEVICE)
                if separate
                else None,
            )
            variables = (query, reference) if separate else (query,)
            expected_gradients = torch.autograd.grad(expected, variables)
            actual_gradients = torch.autograd.grad(actual, variables)
            self.assertGreater(expected.item(), 0)
            torch.testing.assert_close(actual, expected)
            for observed, wanted in zip(
                actual_gradients, expected_gradients, strict=True
            ):
                torch.testing.assert_close(observed, wanted)
