import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.distances import DotProductSimilarity, LpDistance
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    MultiSimilarityLoss,
    NTXentLoss,
    SupConLoss,
)
from pytorch_metric_learning.miners import BatchHardMiner

from .. import TEST_DEVICE, TEST_DTYPES


def contrastive_reference(query, reference, positive_pairs, negative_pairs):
    # Scalar squared distances and analytic gradients, independent of loss/miner code.
    q, r = query.detach().cpu().tolist(), reference.detach().cpu().tolist()
    q_grad = [[0.0] * len(q[0]) for _ in q]
    r_grad = [[0.0] * len(r[0]) for _ in r]
    value = 0.0
    for pairs, positive in [(positive_pairs, True), (negative_pairs, False)]:
        active = []
        for a, b in pairs:
            delta = [x - y for x, y in zip(q[a], r[b])]
            distance = sum(x * x for x in delta)
            loss = distance if positive else 3.0 - distance
            if loss > 0:
                active.append((a, b, delta, loss))
        for a, b, delta, loss in active:
            value += loss / len(active)
            for j, d in enumerate(delta):
                slope = (2.0 if positive else -2.0) * d / len(active)
                q_grad[a][j] += slope
                r_grad[b][j] -= slope
    return tuple(query.new_tensor(x) for x in (value, q_grad, r_grad))


class TestGenericPairLossSinglePair(unittest.TestCase):
    def assert_value_and_gradients(self, actual, query, reference, expected):
        gradients = torch.autograd.grad(actual, (query, reference), allow_unused=True)
        observed = (actual,) + tuple(
            torch.zeros_like(x) if g is None else g
            for x, g in zip((query, reference), gradients)
        )
        for aspect, value, target in zip(
            ("loss", "query_gradient", "reference_gradient"), observed, expected
        ):
            with self.subTest(aspect=aspect):
                self.assertTrue(
                    torch.allclose(value, target, rtol=1e-4, atol=1e-5),
                    msg="{}: expected {}, got {}".format(aspect, target, value),
                )

    def tensors(self, dtype, query, reference):
        return tuple(
            torch.tensor(x, device=TEST_DEVICE, dtype=dtype, requires_grad=True)
            for x in (query, reference)
        )

    def contrastive_loss(self):
        return ContrastiveLoss(
            neg_margin=3, distance=LpDistance(normalize_embeddings=False, power=2)
        )

    def test_reference_batches(self):
        cases = [
            ([[1, 1]], [[3, 1]], [7], [7]),
            ([[1, 1]], [[2, 1]], [7], [9]),
            ([[1, 1]], [[3, 1], [2, 1]], [7], [7, 9]),
            ([[1, 1], [4, 3]], [[3, 1], [2, 1], [4, 4]], [7, 9], [7, 9, 9]),
        ]
        for dtype in TEST_DTYPES:
            for q, r, ql, rl in cases:
                with self.subTest(dtype=dtype, query_labels=ql, reference_labels=rl):
                    query, reference = self.tensors(dtype, q, r)
                    positives, negatives = [], []
                    for a, label in enumerate(ql):
                        for b, ref_label in enumerate(rl):
                            (positives if label == ref_label else negatives).append(
                                (a, b)
                            )
                    actual = self.contrastive_loss()(
                        query,
                        torch.tensor(ql, device=TEST_DEVICE),
                        ref_emb=reference,
                        ref_labels=torch.tensor(rl, device=TEST_DEVICE),
                    )
                    expected = contrastive_reference(
                        query, reference, positives, negatives
                    )
                    self.assert_value_and_gradients(actual, query, reference, expected)

    def test_single_mined_triplet(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                query, reference = self.tensors(dtype, [[1, 1]], [[3, 1], [2, 1]])
                miner = BatchHardMiner(
                    distance=LpDistance(normalize_embeddings=False, power=2)
                )
                triplet = miner(
                    query,
                    torch.tensor([7], device=TEST_DEVICE),
                    reference,
                    torch.tensor([7, 9], device=TEST_DEVICE),
                )
                self.assertEqual([x.tolist() for x in triplet], [[0], [0], [1]])
                actual = self.contrastive_loss()(
                    query, indices_tuple=triplet, ref_emb=reference
                )
                expected = contrastive_reference(query, reference, [(0, 0)], [(0, 1)])
                self.assert_value_and_gradients(actual, query, reference, expected)

    def test_single_prespecified_pair(self):
        for dtype in TEST_DTYPES:
            for positive in (True, False):
                with self.subTest(dtype=dtype, positive=positive):
                    query, reference = self.tensors(
                        dtype, [[1, 1], [4, 3]], [[3, 1], [2, 1], [4, 4]]
                    )
                    empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
                    one = query.new_tensor([1], dtype=torch.long)
                    two = query.new_tensor([2], dtype=torch.long)
                    pairs = (one, two, empty, empty)
                    if not positive:
                        pairs = (empty, empty, one, two)
                    actual = self.contrastive_loss()(
                        query, indices_tuple=pairs, ref_emb=reference
                    )
                    expected = contrastive_reference(
                        query,
                        reference,
                        [(1, 2)] if positive else [],
                        [] if positive else [(1, 2)],
                    )
                    self.assert_value_and_gradients(actual, query, reference, expected)

    def test_single_positive_negative_cross_entropy(self):
        for dtype in TEST_DTYPES:
            for loss_cls in (NTXentLoss, SupConLoss):
                with self.subTest(dtype=dtype, loss=loss_cls.__name__):
                    query, reference = self.tensors(
                        dtype, [[1, 0.5]], [[0, 1], [0.6, 0.8]]
                    )
                    expected = F.cross_entropy(
                        (query @ reference.t()) / 0.5,
                        torch.tensor([0], device=TEST_DEVICE),
                    )
                    gradients = torch.autograd.grad(expected, (query, reference))
                    loss = loss_cls(
                        temperature=0.5,
                        distance=DotProductSimilarity(normalize_embeddings=False),
                    )
                    actual = loss(
                        query,
                        torch.tensor([7], device=TEST_DEVICE),
                        ref_emb=reference,
                        ref_labels=torch.tensor([7, 9], device=TEST_DEVICE),
                    )
                    self.assert_value_and_gradients(
                        actual, query, reference, (expected,) + gradients
                    )

    def test_empty_pairs_still_zero(self):
        for dtype in TEST_DTYPES:
            for loss_cls in (
                ContrastiveLoss,
                NTXentLoss,
                SupConLoss,
                MultiSimilarityLoss,
            ):
                with self.subTest(dtype=dtype, loss=loss_cls.__name__):
                    query, reference = self.tensors(dtype, [[1, 1]], [[3, 1], [2, 1]])
                    empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
                    actual = loss_cls()(
                        query, indices_tuple=(empty,) * 4, ref_emb=reference
                    )
                    self.assert_value_and_gradients(
                        actual,
                        query,
                        reference,
                        (
                            query.new_zeros(()),
                            torch.zeros_like(query),
                            torch.zeros_like(reference),
                        ),
                    )
