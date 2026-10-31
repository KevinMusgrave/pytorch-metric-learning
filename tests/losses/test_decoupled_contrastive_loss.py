import math
import unittest
from functools import partial

import torch

from pytorch_metric_learning import distances, losses, reducers

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def unit(x):
    return x / x.norm(dim=1, keepdim=True)


def two_view_reference(first, second, temperature, sigma=None):
    views = (unit(first), unit(second))
    size = len(first)
    diagonal = [views[0][m] @ views[1][m] for m in range(size)]
    terms = []
    for k in range(2):
        for i in range(size):
            positive = diagonal[i]
            weight = 1.0
            if sigma is not None:
                mean = sum(torch.exp(s / sigma) for s in diagonal) / size
                weight = 2 - torch.exp(positive / sigma) / mean
            negatives = [
                torch.exp(views[k][i] @ views[other][j] / temperature)
                for other in range(2)
                for j in range(size)
                if j != i
            ]
            terms.append(
                -weight * positive / temperature
                + torch.log(torch.stack(negatives).sum())
            )
    return torch.stack(terms).mean()


def labeled_reference(embeddings, labels, temperature, sign=1, distance=None):
    if distance is None:
        distance = unit(embeddings) @ unit(embeddings).T
    scores = sign * distance
    terms = []
    rows = range(len(labels))
    for a in rows:
        for p in rows:
            if a == p or labels[a] != labels[p]:
                continue
            negatives = [
                torch.exp(scores[a, n] / temperature)
                for n in rows
                if labels[a] != labels[n]
            ]
            if negatives:
                terms.append(
                    -scores[a, p] / temperature
                    + torch.log(torch.stack(negatives).sum())
                )
    return torch.stack(terms).mean()


class TestDecoupledContrastiveLoss(unittest.TestCase):
    def views(self, dtype=torch.float64, size=6):
        generator = torch.Generator().manual_seed(11)
        first = torch.randn(size, 5, generator=generator, dtype=torch.float64)
        second = first + 0.5 * torch.randn(
            size, 5, generator=generator, dtype=torch.float64
        )
        return (
            first.to(TEST_DEVICE, dtype).requires_grad_(),
            second.to(TEST_DEVICE, dtype).requires_grad_(),
        )

    def test_equation_6_on_two_views(self):
        for dtype in TEST_DTYPES:
            for temperature in (0.1, 0.5):
                with self.subTest(dtype=dtype, temperature=temperature):
                    loss_func = losses.SelfSupervisedLoss(
                        losses.DecoupledContrastiveLoss(temperature=temperature)
                    )
                    helpers.assert_matches_float64(
                        loss_func,
                        lambda x, y, t=temperature: two_view_reference(x, y, t),
                        self.views(dtype),
                        dtype,
                    )

    def test_equation_7_on_two_views(self):
        for dtype in TEST_DTYPES:
            for sigma in (0.5, 2.0):
                with self.subTest(dtype=dtype, sigma=sigma):
                    loss_func = losses.SelfSupervisedLoss(
                        losses.DecoupledContrastiveLoss(sigma=sigma)
                    )
                    helpers.assert_matches_float64(
                        loss_func,
                        lambda x, y, s=sigma: two_view_reference(x, y, 0.1, s),
                        self.views(dtype),
                        dtype,
                    )

    def test_hand_computed_value(self):
        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64).to(
            TEST_DEVICE
        )
        second = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64).to(
            TEST_DEVICE
        )
        loss_func = losses.SelfSupervisedLoss(
            losses.DecoupledContrastiveLoss(temperature=1.0)
        )
        expected = [
            -1 + math.log(1 + math.e),
            -0 + math.log(2),
            -1 + math.log(1 + math.e),
            -0 + math.log(2 * math.e),
        ]
        self.assertAlmostEqual(
            loss_func(first, second).item(), sum(expected) / 4, places=10
        )

    def test_sum_reducer_matches_the_paper_total(self):
        first, second = self.views()
        loss_func = losses.SelfSupervisedLoss(
            losses.DecoupledContrastiveLoss(reducer=reducers.SumReducer())
        )
        expected = two_view_reference(first, second, 0.1) * 2 * len(first)
        self.assertAlmostEqual(loss_func(first, second).item(), expected.item())

    def test_large_sigma_approaches_dcl(self):
        first, second = self.views()
        plain = losses.SelfSupervisedLoss(losses.DecoupledContrastiveLoss())
        weighted = losses.SelfSupervisedLoss(losses.DecoupledContrastiveLoss(sigma=1e7))
        self.assertAlmostEqual(
            plain(first, second).item(), weighted(first, second).item(), places=4
        )

    def test_weights_favor_hard_positives(self):
        embeddings = torch.tensor(
            [[1.0, 0], [1, 0.1], [1, 2], [0, 1], [-1, 0], [-1, -1]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 3], device=TEST_DEVICE)
        loss_func = losses.DecoupledContrastiveLoss(sigma=0.5)
        terms = loss_func.loss_terms(embeddings, labels)["loss"]["losses"]
        plain = losses.DecoupledContrastiveLoss().loss_terms(embeddings, labels)
        scores = unit(embeddings) @ unit(embeddings).T
        positives = torch.stack(
            [scores[0, 1], scores[1, 0], scores[2, 3], scores[3, 2]]
        )
        weights = 2 - torch.softmax(positives / 0.5, dim=0) * 4
        self.assertAlmostEqual(weights.mean().item(), 1.0)
        self.assertGreater(weights[2].item(), weights[0].item())
        torch.testing.assert_close(
            terms, plain["loss"]["losses"] - (weights - 1) * positives / 0.1
        )

    def test_positive_does_not_scale_the_negative_gradient(self):
        def negative_gradient(loss_func, closeness):
            anchor = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
            positive = torch.tensor([[1.0, closeness]], dtype=torch.float64)
            negative = torch.tensor(
                [[0.2, 1.0]], dtype=torch.float64, requires_grad=True
            )
            rows = torch.cat([anchor, positive, negative]).to(TEST_DEVICE)
            labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
            pairs = (
                torch.tensor([0], device=TEST_DEVICE),
                torch.tensor([1], device=TEST_DEVICE),
                torch.tensor([0], device=TEST_DEVICE),
                torch.tensor([2], device=TEST_DEVICE),
            )
            loss_func(rows, labels, pairs).backward()
            return negative.grad

        decoupled = losses.DecoupledContrastiveLoss(temperature=0.1)
        coupled = losses.NTXentLoss(temperature=0.1)
        torch.testing.assert_close(
            negative_gradient(decoupled, 0.0), negative_gradient(decoupled, 2.0)
        )
        self.assertFalse(
            torch.allclose(
                negative_gradient(coupled, 0.0), negative_gradient(coupled, 2.0)
            )
        )

    def test_multiple_positives_per_anchor(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                first, _ = self.views(dtype, size=8)
                labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=TEST_DEVICE)
                loss_func = losses.DecoupledContrastiveLoss(temperature=0.5)
                helpers.assert_matches_float64(
                    partial(loss_func, labels=labels),
                    partial(labeled_reference, labels=labels, temperature=0.5),
                    (first,),
                    dtype,
                )

    def test_distance_scores_are_negated(self):
        distance = distances.LpDistance()
        first, _ = self.views(size=8)
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=TEST_DEVICE)
        helpers.assert_matches_float64(
            lambda x: losses.DecoupledContrastiveLoss(
                temperature=0.5, distance=distance
            )(x, labels),
            lambda x: labeled_reference(
                x, labels, 0.5, sign=-1, distance=torch.cdist(unit(x), unit(x))
            ),
            (first,),
            torch.float64,
        )

    def test_separate_reference_set(self):
        first, second = self.views()
        labels = torch.arange(6, device=TEST_DEVICE) % 2
        loss = losses.DecoupledContrastiveLoss(temperature=0.5)(
            first, labels, ref_emb=second, ref_labels=labels
        )
        scores = unit(first) @ unit(second).T
        terms = []
        for a in range(6):
            for p in range(6):
                if labels[a] == labels[p]:
                    negatives = [
                        torch.exp(scores[a, n] / 0.5)
                        for n in range(6)
                        if labels[a] != labels[n]
                    ]
                    terms.append(
                        -scores[a, p] / 0.5 + torch.log(torch.stack(negatives).sum())
                    )
        self.assertAlmostEqual(loss.item(), torch.stack(terms).mean().item())

    def test_loss_can_be_negative(self):
        embeddings = torch.tensor(
            [[1.0, 0], [1, 0], [-1, 0], [-1, 0]], device=TEST_DEVICE
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        self.assertLess(
            losses.DecoupledContrastiveLoss()(embeddings, labels).item(), -5
        )

    def test_anchor_without_negatives_is_left_out(self):
        embeddings, _ = self.views(size=4)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        pairs = (
            torch.tensor([0, 2], device=TEST_DEVICE),
            torch.tensor([1, 3], device=TEST_DEVICE),
            torch.tensor([0], device=TEST_DEVICE),
            torch.tensor([2], device=TEST_DEVICE),
        )
        loss_func = losses.DecoupledContrastiveLoss(temperature=0.5)
        scores = unit(embeddings) @ unit(embeddings).T
        expected = -scores[0, 1] / 0.5 + scores[0, 2] / 0.5
        loss = loss_func(embeddings, labels, pairs)
        self.assertAlmostEqual(loss.item(), expected.item())
        loss.backward()
        assert embeddings.grad is not None
        self.assertTrue(embeddings.grad.isfinite().all())

    def test_one_class_and_all_unique_labels_give_a_differentiable_zero(self):
        embeddings, _ = self.views(size=4)
        for labels in (
            torch.zeros(4, dtype=torch.long, device=TEST_DEVICE),
            torch.arange(4, device=TEST_DEVICE),
        ):
            for sigma in (None, 0.5):
                with self.subTest(labels=labels.tolist(), sigma=sigma):
                    loss = losses.DecoupledContrastiveLoss(sigma=sigma)(
                        embeddings, labels
                    )
                    self.assertEqual(loss.item(), 0)
                    loss.backward()

    def test_low_precision_stays_finite_at_small_temperatures(self):
        embeddings = torch.tensor(
            [[1.0, 0.1], [1.0, 0.2], [-1.0, 0.1], [-1.0, 0.2]],
            dtype=torch.float16,
            device=TEST_DEVICE,
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        loss = losses.DecoupledContrastiveLoss(temperature=0.01, sigma=0.01)(
            embeddings, labels
        )
        self.assertTrue(loss.isfinite())
        loss.backward()
        assert embeddings.grad is not None
        self.assertTrue(embeddings.grad.isfinite().all())

    def test_invalid_settings(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    losses.DecoupledContrastiveLoss(temperature=value)
                with self.assertRaisesRegex(ValueError, "sigma"):
                    losses.DecoupledContrastiveLoss(sigma=value)

    def test_anchor_batching_applies_only_without_weights(self):
        self.assertTrue(losses.DecoupledContrastiveLoss().decomposes_by_anchor)
        self.assertFalse(
            losses.DecoupledContrastiveLoss(sigma=0.5).decomposes_by_anchor
        )
