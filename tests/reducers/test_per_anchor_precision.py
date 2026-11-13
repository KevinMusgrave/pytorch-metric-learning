import pytest
import torch

from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    PerAnchorReducer,
    ThresholdReducer,
)

from .. import TEST_DEVICE


def reduce_values(values, anchors, *, aggregation="mean", inner=None):
    embeddings = values.new_zeros((4, 2))
    return PerAnchorReducer(inner, aggregation=aggregation)(
        {"loss": {"losses": values, "indices": anchors}}, embeddings, None
    )


def test_finite_half_loss_does_not_overflow_at_the_per_anchor_sum():
    values = torch.tensor(
        [40000.0, 40000.0, 40000.0],
        dtype=torch.float16,
        requires_grad=True,
        device=TEST_DEVICE,
    )
    actual = reduce_values(
        values, torch.tensor([0, 0, 2], device=TEST_DEVICE), aggregation="sum"
    )
    torch.testing.assert_close(
        actual.double(), torch.tensor(60000.0, dtype=torch.float64, device=TEST_DEVICE)
    )
    actual.backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 0.5))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_inner_threshold_uses_the_unrounded_anchor_mean(dtype):
    step = torch.finfo(dtype).eps
    values = torch.tensor(
        [1.0, 1.0 + step, 2.0], dtype=dtype, requires_grad=True, device=TEST_DEVICE
    )
    actual = reduce_values(
        values,
        torch.tensor([0, 0, 2], device=TEST_DEVICE),
        inner=ThresholdReducer(low=1.0),
    )
    expected = torch.tensor(
        (1 + step / 2 + 2) / 2, dtype=torch.float64, device=TEST_DEVICE
    )
    torch.testing.assert_close(actual.double(), expected)
    actual.backward()
    torch.testing.assert_close(
        values.grad, torch.tensor([0.25, 0.25, 0.5], dtype=dtype, device=TEST_DEVICE)
    )


def test_tiny_positive_anchor_mean_keeps_its_loss_and_gradients():
    smallest = torch.finfo(torch.float16).tiny * torch.finfo(torch.float16).eps
    values = torch.tensor(
        [smallest, 0.0, 0.0, 0.0],
        dtype=torch.float16,
        requires_grad=True,
        device=TEST_DEVICE,
    )
    actual = reduce_values(
        values,
        torch.tensor([0, 0, 0, 2], device=TEST_DEVICE),
        inner=AvgNonZeroReducer(),
    )
    torch.testing.assert_close(
        actual.double(),
        torch.tensor(smallest / 3, dtype=torch.float64, device=TEST_DEVICE),
        rtol=1e-6,
        atol=0,
    )
    actual.backward()
    torch.testing.assert_close(
        values.grad,
        torch.tensor([1 / 3, 1 / 3, 1 / 3, 0], dtype=torch.float16, device=TEST_DEVICE),
    )


def test_contrastive_reference_batch_keeps_large_anchor_sums_finite():
    embeddings = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float16, requires_grad=True, device=TEST_DEVICE
    )
    references = torch.tensor(
        [[0.0, 1.0]] * 50000, dtype=torch.float16, device=TEST_DEVICE
    )
    labels = torch.tensor([0], device=TEST_DEVICE)
    reference_labels = torch.zeros(
        len(references), dtype=torch.long, device=TEST_DEVICE
    )
    loss = ContrastiveLoss(reducer=PerAnchorReducer(aggregation="sum"))
    actual = loss(embeddings, labels, ref_emb=references, ref_labels=reference_labels)
    distance = torch.tensor(2.0, device=TEST_DEVICE).sqrt().half().double()
    torch.testing.assert_close(actual.double(), distance * len(references))
    actual.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_precision_values_dtype_and_gradients_are_preserved(dtype):
    values = torch.tensor(
        [1.0, 3.0, 5.0], dtype=dtype, requires_grad=True, device=TEST_DEVICE
    )
    actual = reduce_values(values, torch.tensor([0, 0, 2], device=TEST_DEVICE))
    torch.testing.assert_close(
        actual, torch.tensor(3.5, dtype=dtype, device=TEST_DEVICE)
    )
    actual.backward()
    torch.testing.assert_close(
        values.grad, torch.tensor([0.25, 0.25, 0.5], dtype=dtype, device=TEST_DEVICE)
    )
