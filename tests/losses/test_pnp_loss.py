import math
import unittest
from functools import partial
from typing import Any, get_args

import torch
import torch.nn
import torch.nn.functional
import torch.nn.functional as F

from pytorch_metric_learning.losses import PNPLoss
from pytorch_metric_learning.types import PNPVariant

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def scalar_sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def scalar_variant(rank, variant, b, alpha):
    if variant == "O":
        return rank, 1.0
    if variant == "Ds":
        return math.log1p(rank), 1.0 / (1.0 + rank)
    if variant == "Iu":
        return (1.0 + rank) * math.log1p(rank), 1.0 + math.log1p(rank)
    if variant == "Ib":
        return (b * rank - math.log1p(b * rank)) / b**2, rank / (1.0 + b * rank)
    return 1.0 - (1.0 + rank) ** (-alpha), alpha * (1.0 + rank) ** (-alpha - 1)


def scalar_pnp_reference(values, labels, temperature, variant, b=2, alpha=4):
    count = len(values)
    norms = [math.sqrt(sum(value * value for value in row)) for row in values]
    units = [
        [value / norm for value in row] for row, norm in zip(values, norms, strict=True)
    ]
    similarities = [
        [
            sum(x * y for x, y in zip(query, reference, strict=True))
            for reference in units
        ]
        for query in units
    ]
    anchors = [i for i in range(count) if labels.count(labels[i]) > 1]
    score_gradient = [[0.0] * count for _ in values]
    loss = 0.0
    for anchor in anchors:
        positives = [i for i in range(count) if labels[i] == labels[anchor]]
        weight = 1.0 / (len(anchors) * len(positives))
        for positive in positives:
            negatives = [i for i in range(count) if labels[i] != labels[positive]]
            probabilities = [
                scalar_sigmoid(
                    (similarities[anchor][negative] - similarities[anchor][positive])
                    / temperature
                )
                for negative in negatives
            ]
            value, slope = scalar_variant(sum(probabilities), variant, b, alpha)
            loss += weight * value
            for negative, probability in zip(negatives, probabilities, strict=True):
                coefficient = (
                    weight * slope * probability * (1.0 - probability) / temperature
                )
                score_gradient[anchor][negative] += coefficient
                score_gradient[anchor][positive] -= coefficient
    gradient = [[0.0] * len(row) for row in values]
    for query in range(count):
        for reference in range(count):
            coefficient = score_gradient[query][reference]
            similarity = similarities[query][reference]
            for feature in range(len(values[query])):
                gradient[query][feature] += (
                    coefficient
                    * (units[reference][feature] - similarity * units[query][feature])
                    / norms[query]
                )
                gradient[reference][feature] += (
                    coefficient
                    * (units[query][feature] - similarity * units[reference][feature])
                    / norms[reference]
                )
    return loss, gradient


class OriginalImplementationPNP(torch.nn.Module):
    def __init__(self, b, alpha, anneal, variant, bs, classes):
        super().__init__()
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant
        self.batch_size = bs
        self.num_id = classes
        self.samples_per_class = int(bs / classes)

        mask = 1.0 - torch.eye(self.batch_size)
        for i in range(self.num_id):
            mask[
                i * (self.samples_per_class) : (i + 1) * (self.samples_per_class),
                i * (self.samples_per_class) : (i + 1) * (self.samples_per_class),
            ] = 0

        self.mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)

    def forward(self, batch):
        dtype, device = batch.dtype, batch.device
        self.mask = self.mask.type(dtype).to(device)

        sim_all = self.compute_aff(batch)

        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * self.mask
        sim_all_rk = torch.sum(sim_sg, dim=-1)
        if self.variant == "PNP-D_s":
            sim_all_rk = torch.log(1 + sim_all_rk)
        elif self.variant == "PNP-D_q":
            sim_all_rk = 1 / (1 + sim_all_rk) ** (self.alpha)

        elif self.variant == "PNP-I_u":
            sim_all_rk = (1 + sim_all_rk) * torch.log(1 + sim_all_rk)

        elif self.variant == "PNP-I_b":
            b = self.b
            sim_all_rk = 1 / b**2 * (b * sim_all_rk - torch.log(1 + b * sim_all_rk))
        elif self.variant == "PNP-O":
            pass
        else:
            raise ValueError(f"variantation <{self.variant}> not available!")

        loss = torch.zeros(1).type(dtype).to(device)
        group = int(self.batch_size / self.num_id)

        for ind in range(self.num_id):
            neg_divide = torch.sum(
                sim_all_rk[
                    (ind * group) : ((ind + 1) * group),
                    (ind * group) : ((ind + 1) * group),
                ]
                / group
            )
            loss = loss + (neg_divide / self.batch_size)
        if self.variant == "PNP-D_q":
            return 1 - loss
        else:
            return loss

    def sigmoid(self, tensor, temp=1.0):
        exponent = -tensor / temp
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def compute_aff(self, x):
        return torch.mm(x, x.t())


class TestPNPLoss(unittest.TestCase):
    def test_pnp_loss(self):
        torch.manual_seed(30293)
        bs = 180
        classes = 30
        for variant in get_args(PNPVariant.__value__):
            original_variant = {
                "Ds": "PNP-D_s",
                "Dq": "PNP-D_q",
                "Iu": "PNP-I_u",
                "Ib": "PNP-I_b",
                "O": "PNP-O",
            }[variant]
            b, alpha, anneal = 2, 4, 0.01
            loss_func = PNPLoss(b, alpha, anneal, variant)
            original_loss_func = OriginalImplementationPNP(
                b, alpha, anneal, original_variant, bs, classes
            ).to(TEST_DEVICE)

            for dtype in TEST_DTYPES:
                embeddings = torch.randn(
                    180, 32, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                labels = (
                    torch.tensor([[i] * (int(bs / classes)) for i in range(classes)])
                    .reshape(-1)
                    .to(TEST_DEVICE)
                )
                loss = loss_func(embeddings, labels)
                loss.backward()
                correct_loss = original_loss_func(F.normalize(embeddings, dim=-1))

                rtol = helpers.rtol(dtype)
                self.assertTrue(torch.isclose(loss, correct_loss[0], rtol=rtol))

        invalid: Any = "PNP"
        with self.assertRaises(ValueError):
            PNPLoss(b, alpha, anneal, invalid)

    def test_negatives_that_have_no_positives(self):
        loss_func = PNPLoss()
        labels = torch.tensor([1, 1, 2], device=TEST_DEVICE)
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(3, 32, dtype=dtype, device=TEST_DEVICE)
            loss = loss_func(embeddings, labels)
            self.assertTrue(not torch.isnan(loss))

    def test_anchors_without_other_positives_are_excluded(self):
        labels = torch.tensor([1, 1, 2, 1, 3], device=TEST_DEVICE)
        torch.manual_seed(4)
        embeddings = torch.randn(5, 6)

        def reference(x, variant, alpha):
            x = F.normalize(x, dim=1)
            sims = x @ x.T
            per_anchor = []
            for q in range(len(labels)):
                positives = [i for i in range(len(labels)) if labels[i] == labels[q]]
                if len(positives) == 1:
                    continue
                negatives = [k for k in range(len(labels)) if labels[k] != labels[q]]
                ranks = [
                    torch.sigmoid((sims[q, negatives] - sims[q, i]) / 0.1).sum()
                    for i in positives
                ]
                if variant == "Dq":
                    ranks = [1 / (1 + r) ** alpha for r in ranks]
                per_anchor.append(torch.stack(ranks).mean())
            loss = torch.stack(per_anchor).mean()
            return 1 - loss if variant == "Dq" else loss

        for variant in ("O", "Dq"):
            loss_func = PNPLoss(alpha=2, anneal=0.1, variant=variant)
            for dtype in TEST_DTYPES:
                with self.subTest(variant=variant, dtype=dtype):
                    helpers.assert_matches_float64(
                        partial(loss_func, labels=labels),
                        partial(reference, variant=variant, alpha=2),
                        [embeddings],
                        dtype,
                    )

    def test_batches_without_positive_pairs_give_differentiable_zero(self):
        for labels in ([], [0], [0, 1]):
            for dtype in TEST_DTYPES:
                with self.subTest(labels=labels, dtype=dtype):
                    embeddings = torch.randn(
                        len(labels), 4, dtype=dtype, device=TEST_DEVICE
                    ).requires_grad_()
                    loss = PNPLoss()(
                        embeddings, torch.tensor(labels, device=TEST_DEVICE).long()
                    )
                    loss.backward()
                    self.assertEqual(loss.item(), 0)
                    self.assertIsNotNone(embeddings.grad)


class TestPNPScalarReference(unittest.TestCase):
    unit_vectors = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0.5, 0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5, 0.5),
    )

    half_overflow_vectors = ((1, 0), (0.8, 0.6), (0.6, 0.8), (0, 1))

    def embeddings(self, dtype, vectors=unit_vectors):
        return torch.tensor(
            vectors, dtype=dtype, device=TEST_DEVICE, requires_grad=True
        )

    def test_variants_match_scalar_values_and_analytic_gradients(self):
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        tolerances = {torch.float16: 3e-3, torch.float32: 1e-4, torch.float64: 2e-11}
        for dtype in TEST_DTYPES:
            for variant in get_args(PNPVariant.__value__):
                for temperature in (0.01, 0.5):
                    with self.subTest(
                        dtype=dtype, variant=variant, temperature=temperature
                    ):
                        embeddings = self.embeddings(dtype)
                        expected, expected_gradient = scalar_pnp_reference(
                            embeddings.detach().double().tolist(),
                            labels.tolist(),
                            temperature,
                            variant,
                        )
                        loss = PNPLoss(
                            b=2, alpha=4, anneal=temperature, variant=variant
                        )(embeddings, labels)
                        (gradient,) = torch.autograd.grad(loss, embeddings)
                        expected = torch.tensor(
                            expected, dtype=torch.float64, device=TEST_DEVICE
                        )
                        expected_gradient = torch.tensor(
                            expected_gradient, dtype=torch.float64, device=TEST_DEVICE
                        )
                        self.assertGreater(expected_gradient.abs().max().item(), 1e-3)
                        self.assertTrue(torch.isfinite(gradient).all())
                        tolerance = tolerances[dtype]
                        self.assertTrue(
                            torch.allclose(
                                loss.double(), expected, rtol=tolerance, atol=tolerance
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                gradient.double(),
                                expected_gradient,
                                rtol=tolerance,
                                atol=tolerance,
                            )
                        )

    def test_masked_comparisons_keep_backward_finite(self):
        labels = torch.zeros(4, dtype=torch.long, device=TEST_DEVICE)
        for vectors in (self.unit_vectors, self.half_overflow_vectors):
            for dtype in TEST_DTYPES:
                for variant in get_args(PNPVariant.__value__):
                    with self.subTest(vectors=vectors, dtype=dtype, variant=variant):
                        embeddings = self.embeddings(dtype, vectors)
                        loss = PNPLoss(b=2, alpha=4, variant=variant)(
                            embeddings, labels
                        )
                        (gradient,) = torch.autograd.grad(loss, embeddings)
                        self.assertEqual(loss.item(), 0)
                        self.assertTrue(
                            torch.equal(gradient, torch.zeros_like(gradient))
                        )

    def test_half_exponent_overflow_keeps_default_temperature_finite(self):
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        for variant in get_args(PNPVariant.__value__):
            with self.subTest(variant=variant):
                embeddings = self.embeddings(torch.float16, self.half_overflow_vectors)
                expected, _ = scalar_pnp_reference(
                    embeddings.detach().double().tolist(),
                    labels.tolist(),
                    0.01,
                    variant,
                )
                loss = PNPLoss(b=2, alpha=4, variant=variant)(embeddings, labels)
                (gradient,) = torch.autograd.grad(loss, embeddings)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().max().item(), 0)
                self.assertAlmostEqual(loss.item(), expected, delta=0.02)

    def test_negative_tail_keeps_representable_values_and_gradients(self):
        temperature = 0.01
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        tolerances = {torch.float32: 5e-5, torch.float64: 1e-10}
        for dtype, tolerance in tolerances.items():
            for gap in (45, 46, 48, 49):
                with self.subTest(dtype=dtype, gap=gap):
                    c = 1 - gap * temperature
                    y_value = math.sqrt(1 - c * c)
                    embeddings = torch.tensor(
                        [[1, 0], [1, 0], [c, y_value], [c, y_value]],
                        dtype=dtype,
                        device=TEST_DEVICE,
                        requires_grad=True,
                    )
                    x, y = embeddings[2].detach().double().tolist()
                    norm = math.hypot(x, y)
                    probability = scalar_sigmoid((x / norm - 1) / temperature)
                    expected = torch.tensor(
                        [
                            2 * probability,
                            probability * (1 - probability) / temperature * y / norm,
                        ],
                        dtype=torch.float64,
                        device=TEST_DEVICE,
                    )
                    loss = PNPLoss(anneal=temperature)(embeddings, labels)
                    (gradient,) = torch.autograd.grad(loss, embeddings)
                    actual = torch.stack([loss, gradient[0, 1]]).double()
                    self.assertTrue(torch.all(actual > 0))
                    self.assertTrue(
                        torch.allclose(actual, expected, rtol=tolerance, atol=0)
                    )
