"""Independent scalar PNP objectives and derivatives, including masked half tails."""

import math
import unittest

import torch

from pytorch_metric_learning.losses import PNPLoss

from .. import TEST_DEVICE


def scalar_sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def scalar_pnp_reference(values, labels, temperature, variant, b=2, alpha=4):
    """Python scalar rank sums, analytic variant slopes, and cosine chain rule."""
    norms = [math.sqrt(sum(value * value for value in row)) for row in values]
    units = [[value / norm for value in row] for row, norm in zip(values, norms)]
    similarities = [
        [
            sum(left * right for left, right in zip(query, reference))
            for reference in units
        ]
        for query in units
    ]
    safe = [anchor for anchor in range(len(labels)) if labels.count(labels[anchor]) > 1]
    score_gradient = [[0.0] * len(values) for _ in values]
    loss = 0.0
    for anchor in safe:
        positives = [
            positive
            for positive in range(len(labels))
            if labels[positive] == labels[anchor]
        ]
        for positive in positives:
            negatives = [
                negative
                for negative in range(len(labels))
                if labels[negative] != labels[positive]
            ]
            comparisons = []
            for negative in negatives:
                gap = (
                    similarities[anchor][negative] - similarities[anchor][positive]
                ) / temperature
                probability = scalar_sigmoid(max(-50.0, min(50.0, gap)))
                # Values at the clamp boundary have exponentially tiny derivatives;
                # choose the existing torch.clamp boundary subgradient (one).
                derivative = (
                    probability * (1.0 - probability) / temperature
                    if abs(gap) <= 50
                    else 0.0
                )
                comparisons.append((negative, probability, derivative))
            rank = sum(probability for _, probability, _ in comparisons)
            if variant == "O":
                value, slope = rank, 1.0
            elif variant == "Ds":
                value, slope = math.log1p(rank), 1.0 / (1.0 + rank)
            elif variant == "Iu":
                value, slope = (1.0 + rank) * math.log1p(rank), 1.0 + math.log1p(rank)
            elif variant == "Ib":
                value, slope = (b * rank - math.log1p(b * rank)) / b**2, rank / (
                    1.0 + b * rank
                )
            else:
                assert variant == "Dq"
                value, slope = 1.0 - (1.0 + rank) ** (-alpha), alpha * (1.0 + rank) ** (
                    -alpha - 1
                )
            weight = 1.0 / (len(safe) * len(positives))
            loss += weight * value
            for negative, _, derivative in comparisons:
                coefficient = weight * slope * derivative
                score_gradient[anchor][negative] += coefficient
                score_gradient[anchor][positive] -= coefficient
    gradient = [[0.0] * len(row) for row in values]
    for query in range(len(values)):
        for reference in range(len(values)):
            coefficient = score_gradient[query][reference]
            for feature in range(len(values[query])):
                gradient[query][feature] += (
                    coefficient
                    * (
                        units[reference][feature]
                        - similarities[query][reference] * units[query][feature]
                    )
                    / norms[query]
                )
                gradient[reference][feature] += (
                    coefficient
                    * (
                        units[query][feature]
                        - similarities[query][reference] * units[reference][feature]
                    )
                    / norms[reference]
                )
    return loss, gradient


class TestPNPSigmoidStability(unittest.TestCase):
    variants = ("Ds", "Dq", "Iu", "Ib", "O")
    # Every row has exact binary unit norm, so half normalization and dot products
    # do not contaminate the independent default-temperature derivative oracle.
    unit_vectors = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0.5, 0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5, 0.5),
    )

    def require_half_backend(self, dtype, public_forward=False):
        if dtype != torch.float16 or TEST_DEVICE.type != "cpu":
            return
        operations = []
        if public_forward:
            operations.append(
                (
                    "CPU half matrix multiply forward/backward",
                    lambda value: torch.mm(value, value),
                    {
                        "\"addmm\" not implemented for 'Half'",
                        "\"addmm_impl_cpu_\" not implemented for 'Half'",
                    },
                )
            )
        operations.append(
            (
                "CPU half sigmoid forward/backward",
                torch.sigmoid,
                {
                    "\"sigmoid_cpu\" not implemented for 'Half'",
                    "\"sigmoid_backward_cpu\" not implemented for 'Half'",
                },
            )
        )
        for operation, compute, unsupported in operations:
            value = torch.ones(
                (1, 1), dtype=dtype, device=TEST_DEVICE, requires_grad=True
            )
            try:
                result = compute(value)
                result.sum().backward()
            except RuntimeError as error:
                if str(error) in unsupported:
                    self.skipTest("{}: {}".format(operation, error))
                raise
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_half_sigmoid_overflow_retains_finite_values_and_gradients(self):
        self.require_half_backend(torch.float16)
        values = torch.tensor(
            [-0.45, -0.2, -0.12, -0.03, 0.0, 0.03, 0.12, 0.2, 0.45],
            dtype=torch.float16,
            device=TEST_DEVICE,
            requires_grad=True,
        )
        probabilities = [
            scalar_sigmoid(value / 0.01)
            for value in values.detach().double().cpu().tolist()
        ]
        expected_gradient = [value * (1.0 - value) / 0.01 for value in probabilities]
        actual = PNPLoss().sigmoid(values, temp=0.01)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(
            torch.allclose(
                actual.double(),
                torch.tensor(probabilities, dtype=torch.float64, device=TEST_DEVICE),
                rtol=3e-3,
                atol=1e-3,
            )
        )
        # Positive half tails round to one; below one half ulp their derivative can
        # round to zero. Near zero the derivative remains substantial and checked.
        self.assertTrue(
            torch.allclose(
                values.grad.double(),
                torch.tensor(
                    expected_gradient, dtype=torch.float64, device=TEST_DEVICE
                ),
                rtol=3e-3,
                atol=1e-3,
            )
        )
        self.assertEqual(values.grad[4].item(), 25.0)

    def test_negative_float32_tail_preserves_representable_derivative(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                values = torch.tensor(
                    [-0.45, -0.46, -0.48, -0.49],
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                expected = [
                    scalar_sigmoid(value / 0.01)
                    for value in values.detach().double().cpu().tolist()
                ]
                expected_gradient = [value * (1.0 - value) / 0.01 for value in expected]
                actual = PNPLoss().sigmoid(values, temp=0.01)
                actual.sum().backward()
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertTrue((values.grad > 0).all())
                tolerance = 5e-5 if dtype == torch.float32 else 1e-12
                self.assertTrue(
                    torch.allclose(
                        actual.double(),
                        torch.tensor(expected, dtype=torch.float64, device=TEST_DEVICE),
                        rtol=tolerance,
                        atol=0.0,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        values.grad.double(),
                        torch.tensor(
                            expected_gradient, dtype=torch.float64, device=TEST_DEVICE
                        ),
                        rtol=tolerance,
                        atol=0.0,
                    )
                )

    def test_public_five_variants_value_and_analytic_embedding_gradients(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            for variant in self.variants:
                for temperature in (0.01, 0.5):
                    with self.subTest(
                        dtype=dtype, variant=variant, temperature=temperature
                    ):
                        self.require_half_backend(dtype, public_forward=True)
                        embeddings = torch.tensor(
                            self.unit_vectors,
                            dtype=dtype,
                            device=TEST_DEVICE,
                            requires_grad=True,
                        )
                        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
                        expected_value, expected_gradient = scalar_pnp_reference(
                            embeddings.detach().double().cpu().tolist(),
                            labels.tolist(),
                            temperature,
                            variant,
                        )
                        actual = PNPLoss(
                            b=2, alpha=4, anneal=temperature, variant=variant
                        )(embeddings, labels)
                        actual.backward()
                        self.assertTrue(torch.isfinite(actual))
                        self.assertTrue(torch.isfinite(embeddings.grad).all())
                        self.assertTrue(
                            any(
                                abs(value) > 1e-3
                                for row in expected_gradient
                                for value in row
                            )
                        )
                        tolerance = (
                            3e-3
                            if dtype == torch.float16
                            else 1e-4 if dtype == torch.float32 else 2e-11
                        )
                        self.assertTrue(
                            torch.allclose(
                                actual.double(),
                                torch.tensor(
                                    expected_value,
                                    dtype=torch.float64,
                                    device=TEST_DEVICE,
                                ),
                                rtol=tolerance,
                                atol=tolerance,
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                embeddings.grad.double(),
                                torch.tensor(
                                    expected_gradient,
                                    dtype=torch.float64,
                                    device=TEST_DEVICE,
                                ),
                                rtol=tolerance,
                                atol=tolerance,
                            )
                        )

    def test_masked_zero_negatives_keep_public_backward_finite(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            for variant in self.variants:
                with self.subTest(dtype=dtype, variant=variant):
                    self.require_half_backend(dtype, public_forward=True)
                    embeddings = torch.tensor(
                        self.unit_vectors,
                        dtype=dtype,
                        device=TEST_DEVICE,
                        requires_grad=True,
                    )
                    # All four anchors have real positives, including each other:
                    # this does not hit the no-safe-positive early zero return.
                    labels = torch.zeros(4, dtype=torch.long, device=TEST_DEVICE)
                    actual = PNPLoss(b=2, alpha=4, variant=variant)(embeddings, labels)
                    actual.backward()
                    self.assertEqual(actual.item(), 0.0)
                    self.assertTrue(torch.isfinite(embeddings.grad).all())
                    self.assertTrue(
                        torch.equal(embeddings.grad, torch.zeros_like(embeddings.grad))
                    )

    def test_clamping_preserves_values_and_zero_outside_derivatives(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                values = torch.tensor(
                    [-0.75, -0.03, 0.0, 0.03, 0.75],
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                scalars = values.detach().double().cpu().tolist()
                expected = [
                    scalar_sigmoid(max(-50.0, min(50.0, value / 0.01)))
                    for value in scalars
                ]
                expected_gradient = [
                    (
                        probability * (1.0 - probability) / 0.01
                        if abs(value / 0.01) < 50
                        else 0.0
                    )
                    for value, probability in zip(scalars, expected)
                ]
                actual = PNPLoss().sigmoid(values, temp=0.01)
                actual.sum().backward()
                tolerance = 5e-5 if dtype == torch.float32 else 1e-12
                self.assertTrue(
                    torch.allclose(
                        actual.double(),
                        torch.tensor(expected, dtype=torch.float64, device=TEST_DEVICE),
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        values.grad.double(),
                        torch.tensor(
                            expected_gradient, dtype=torch.float64, device=TEST_DEVICE
                        ),
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
                self.assertEqual(values.grad[0].item(), 0.0)
                self.assertEqual(values.grad[-1].item(), 0.0)
