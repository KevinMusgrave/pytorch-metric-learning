import math
from decimal import Decimal, localcontext

import pytest
import torch

from pytorch_metric_learning.evaluation import (
    AccuracyCalculator,
    RetrievalInputs,
    ndcg,
    ndcg_metric,
)

from .. import TEST_DEVICE


def decimal_ndcg(gains, order, depth, omit=None):
    with localcontext(prec=100):
        values = [Decimal.from_float(value) for value in gains]
        retained = [value for index, value in enumerate(values) if index != omit]
        discounts = [Decimal(2).ln() / Decimal(rank + 2).ln() for rank in range(depth)]
        numerator = sum(
            (
                values[index] * discount
                for index, discount in zip(order[:depth], discounts, strict=True)
            ),
            start=Decimal(0),
        )
        denominator = sum(
            (
                value * discount
                for value, discount in zip(
                    sorted(retained, reverse=True)[:depth], discounts, strict=True
                )
            ),
            start=Decimal(0),
        )
        return float(numerator / denominator)


@pytest.mark.parametrize("scale", [1.0, 1e308, 1e-320])
@pytest.mark.parametrize("order", [[0, 1, 2], [1, 0, 2], [2, 1, 0]])
@pytest.mark.parametrize("depth", [None, 1, 2])
def test_ndcg_preserves_ratio_across_finite_gain_ranges(scale, order, depth):
    gains = (
        torch.tensor([1.5, 1.3, 1.0], device=TEST_DEVICE, dtype=torch.float64) * scale
    )
    labels = torch.tensor([0], device=TEST_DEVICE)
    neighbors = torch.tensor([order], device=TEST_DEVICE)
    inputs = RetrievalInputs(
        labels,
        neighbors,
        labels[:, None] == neighbors,
        torch.tensor([1], device=TEST_DEVICE),
        torch.arange(3, device=TEST_DEVICE),
        False,
    )
    actual = ndcg(inputs, lambda query, references: gains[references], k=depth)
    expected = decimal_ndcg(gains.cpu().tolist(), order, 3 if depth is None else depth)
    torch.testing.assert_close(
        actual, actual.new_tensor([expected]), rtol=1e-14, atol=1e-14
    )
    assert actual.dtype == torch.float64


@pytest.mark.parametrize("scale", [1.0, 1e308, 1e-320])
def test_ndcg_excludes_self_before_choosing_ideal_gain_scale(scale):
    gains = (
        torch.tensor([1.5, 1.3, 1.0], device=TEST_DEVICE, dtype=torch.float64) * scale
    )
    inputs = RetrievalInputs(
        torch.tensor([0], device=TEST_DEVICE),
        torch.tensor([[2, 1]], device=TEST_DEVICE),
        torch.zeros(1, 2, device=TEST_DEVICE, dtype=torch.bool),
        torch.tensor([0], device=TEST_DEVICE),
        torch.arange(3, device=TEST_DEVICE),
        True,
    )
    actual = ndcg(inputs, lambda query, references: gains[references])
    expected = decimal_ndcg(gains.cpu().tolist(), [2, 1], 2, omit=0)
    torch.testing.assert_close(
        actual, actual.new_tensor([expected]), rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("scale", [1e308, 1e-320])
@pytest.mark.parametrize("depth", [None, 2])
def test_registered_ndcg_retains_pooled_and_per_class_results(scale, depth):
    gains = (
        torch.tensor([1.5, 1.3, 1.0], device=TEST_DEVICE, dtype=torch.float64) * scale
    )
    order = torch.tensor([[1, 0, 2], [2, 1, 0]], device=TEST_DEVICE)

    def retrieve(query, k, reference, ref_includes_query):
        return torch.zeros(len(query), k, device=query.device), order[:, :k]

    model = AccuracyCalculator(
        include=("graded",),
        device=TEST_DEVICE,
        knn_func=retrieve,
        custom_metrics={
            "graded": ndcg_metric(lambda query, references: gains[references], k=depth)
        },
    )
    query = torch.zeros(2, 1, device=TEST_DEVICE)
    labels = torch.tensor([0, 1], device=TEST_DEVICE)
    reference = torch.zeros(3, 1, device=TEST_DEVICE)
    ref_labels = torch.arange(3, device=TEST_DEVICE)
    expected = [
        decimal_ndcg(gains.cpu().tolist(), row, 3 if depth is None else depth)
        for row in order.cpu().tolist()
    ]
    actual = model.get_accuracy(query, labels, reference, ref_labels)
    per_class = model.get_accuracy_per_class(query, labels, reference, ref_labels)
    assert actual["graded"] == pytest.approx(sum(expected) / 2, rel=1e-14, abs=1e-14)
    assert per_class["graded"] == pytest.approx(
        dict(enumerate(expected)), rel=1e-14, abs=1e-14
    )


@pytest.mark.parametrize("size", [0, 3])
def test_ndcg_zero_ideal_gain_remains_undefined(size):
    inputs = RetrievalInputs(
        torch.tensor([0], device=TEST_DEVICE),
        torch.zeros(1, size, device=TEST_DEVICE, dtype=torch.long),
        torch.zeros(1, size, device=TEST_DEVICE, dtype=torch.bool),
        torch.tensor([0], device=TEST_DEVICE),
        torch.arange(size, device=TEST_DEVICE),
        False,
    )
    actual = ndcg(
        inputs,
        lambda query, references: torch.zeros_like(references, dtype=torch.float64),
    )
    assert math.isnan(actual.item())
