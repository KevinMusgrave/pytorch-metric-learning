import math
from decimal import Decimal, localcontext

import pytest
import torch

from pytorch_metric_learning import distances, losses, reducers

from .. import TEST_DEVICE


def decimal_decoupled(
    matrix, labels, reference_labels, temperature, sigma, inverted, self_search
):
    with localcontext(prec=100):
        sign = Decimal(1 if inverted else -1)
        scores = [[sign * Decimal.from_float(x) for x in row] for row in matrix]
        temperature = Decimal.from_float(temperature)
        selected = [
            (a, p)
            for a, label in enumerate(labels)
            for p, reference in enumerate(reference_labels)
            if label == reference
            and (not self_search or a != p)
            and any(other != label for other in reference_labels)
        ]
        positives = [scores[a][p] for a, p in selected]
        weights = [Decimal(1) for _ in selected]
        if sigma is not None:
            sigma = Decimal.from_float(sigma)
            exponentials = [((p - max(positives)) / sigma).exp() for p in positives]
            weights = [
                value * len(selected) / sum(exponentials) for value in exponentials
            ]
        weighted_mean = sum(
            (
                weight * positive
                for weight, positive in zip(weights, positives, strict=True)
            ),
            start=Decimal(0),
        ) / len(selected)
        gradient = [[Decimal(0) for _ in row] for row in matrix]
        values = []
        for (anchor, positive), weight in zip(selected, weights, strict=True):
            positive_score = scores[anchor][positive]
            negatives = [
                index
                for index, label in enumerate(reference_labels)
                if label != labels[anchor]
            ]
            exponentials = [
                ((scores[anchor][negative] - positive_score) / temperature).exp()
                for negative in negatives
            ]
            total = sum(exponentials, start=Decimal(0))
            values.append(total.ln() + (weight - 1) * positive_score / temperature)
            for negative, exponential in zip(negatives, exponentials, strict=True):
                gradient[anchor][negative] += (
                    sign * exponential / (temperature * total * len(selected))
                )
            derivative = (weight - 2) / temperature
            if sigma is not None:
                derivative += (
                    weight * (positive_score - weighted_mean) / (sigma * temperature)
                )
            gradient[anchor][positive] += sign * derivative / len(selected)
        return float(sum(values) / len(values)), [
            [float(x) for x in row] for row in gradient
        ]


@pytest.mark.parametrize(
    "dtype,offset",
    [
        (torch.float16, 2000.0),
        (torch.bfloat16, 10000.0),
        (torch.float32, 1e8),
        (torch.float64, 1e16),
    ],
)
@pytest.mark.parametrize("sigma", [None, 0.5])
@pytest.mark.parametrize("inverted", [True, False])
@pytest.mark.parametrize("reducer", ["mean", "sum"])
def test_decoupled_equal_scores_keep_log_negative_count(
    dtype, offset, sigma, inverted, reducer
):
    matrix = torch.full(
        (4, 4), offset, device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    metric = (
        distances.DotProductSimilarity(normalize_embeddings=False)
        if inverted
        else distances.LpDistance(normalize_embeddings=False)
    )
    reduction = reducers.MeanReducer() if reducer == "mean" else reducers.SumReducer()
    model = losses.DecoupledContrastiveLoss(
        temperature=1, sigma=sigma, distance=metric, reducer=reduction
    )
    actual = model(
        torch.zeros(4, 2, device=TEST_DEVICE, dtype=dtype), labels, mat=matrix
    )
    actual_gradient = torch.autograd.grad(actual, matrix)[0]
    expected, expected_gradient = decimal_decoupled(
        matrix.detach().double().cpu().tolist(),
        labels.tolist(),
        labels.tolist(),
        1.0,
        sigma,
        inverted,
        True,
    )
    multiplier = 1 if reducer == "mean" else 4
    tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 3e-6
    torch.testing.assert_close(
        actual, actual.new_tensor(expected * multiplier), rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        actual_gradient,
        actual_gradient.new_tensor(expected_gradient) * multiplier,
        rtol=tolerance,
        atol=0,
    )


@pytest.mark.parametrize("sigma", [None, 0.5, 3.0])
@pytest.mark.parametrize("inverted", [True, False])
def test_decoupled_weighted_rectangular_objective_keeps_absolute_positive_scores(
    sigma, inverted
):
    matrix = torch.tensor(
        [
            [0.2, 0.5, -0.3, 0.7, 0.1],
            [0.5, 0.25, 0.3, 0.125, -0.25],
            [0, 0.1, 0.2, 0.3, 0.4],
        ],
        device=TEST_DEVICE,
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2], device=TEST_DEVICE)
    ref_labels = torch.tensor([0, 0, 1, 1, 3], device=TEST_DEVICE)
    metric = (
        distances.DotProductSimilarity(normalize_embeddings=False)
        if inverted
        else distances.LpDistance(normalize_embeddings=False)
    )
    model = losses.DecoupledContrastiveLoss(
        temperature=0.25, sigma=sigma, distance=metric
    )
    actual = model(
        torch.zeros(3, 2, device=TEST_DEVICE, dtype=torch.float64),
        labels,
        ref_emb=torch.zeros(5, 2, device=TEST_DEVICE, dtype=torch.float64),
        ref_labels=ref_labels,
        mat=matrix,
    )
    gradient = torch.autograd.grad(actual, matrix)[0]
    expected, expected_gradient = decimal_decoupled(
        matrix.detach().cpu().tolist(),
        labels.tolist(),
        ref_labels.tolist(),
        0.25,
        sigma,
        inverted,
        False,
    )
    torch.testing.assert_close(
        actual, actual.new_tensor(expected), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        gradient, gradient.new_tensor(expected_gradient), rtol=1e-12, atol=1e-12
    )


def test_decoupled_finite_temperature_overflow_does_not_drop_anchors():
    matrix = torch.full((4, 4), 1e30, device=TEST_DEVICE, requires_grad=True)
    model = losses.DecoupledContrastiveLoss(
        temperature=1e-10,
        distance=distances.DotProductSimilarity(normalize_embeddings=False),
    )
    actual = model(
        torch.zeros(4, 2, device=TEST_DEVICE),
        torch.tensor([0, 0, 1, 1], device=TEST_DEVICE),
        mat=matrix,
    )
    torch.testing.assert_close(actual, actual.new_tensor(math.log(2)))
    gradient = torch.autograd.grad(actual, matrix)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


@pytest.mark.parametrize("sigma", [None, 0.5])
def test_decoupled_positive_pairs_without_negatives_keep_differentiable_zero(sigma):
    query = torch.eye(3, device=TEST_DEVICE, requires_grad=True)
    actual = losses.DecoupledContrastiveLoss(sigma=sigma)(
        query, torch.zeros(3, dtype=torch.long, device=TEST_DEVICE)
    )
    actual.backward()
    assert actual.item() == 0
    torch.testing.assert_close(query.grad, torch.zeros_like(query))
