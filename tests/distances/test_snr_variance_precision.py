import pytest
import torch

from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.losses import SignalToNoiseRatioContrastiveLoss
from pytorch_metric_learning.types import PairIndices

from .. import TEST_DEVICE
from ..helpers import assert_close

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def rows(dtype):
    result = torch.ones(4, 512, dtype=dtype, device=TEST_DEVICE)
    for i in range(4):
        result[i, i] = 1.03125 + i / 128
    return result


def reference_rows(x, dtype, normalize):
    if not normalize:
        return x
    normalized = x / x.square().sum(dim=1, keepdim=True).sqrt()
    return normalized.to(dtype).double()


def reference_ratio(query, reference):
    centered_query = query - query.mean()
    difference = query - reference
    centered_difference = difference - difference.mean()
    return centered_difference.square().sum() / centered_query.square().sum()


def reference_matrix(query, reference):
    return torch.stack(
        [torch.stack([reference_ratio(q, r) for r in reference]) for q in query]
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normalize", [True, False])
def test_small_variance_matrix_and_pairwise_match_float64(dtype, normalize):
    points = rows(dtype)
    query = points[:2].clone().requires_grad_()
    reference = points[2:].clone().requires_grad_()
    exact_query = query.detach().double().requires_grad_()
    exact_reference = reference.detach().double().requires_grad_()
    distance = SNRDistance(normalize_embeddings=normalize)
    expected = reference_matrix(
        reference_rows(exact_query, dtype, normalize),
        reference_rows(exact_reference, dtype, normalize),
    )
    actual = distance(query, reference)
    expected_dtype = dtype if normalize or dtype == torch.float64 else torch.float32
    assert actual.dtype == expected_dtype
    assert actual.isfinite().all()
    assert_close(actual, expected, dtype)
    assert_close(distance.pairwise(query, reference), expected.diagonal(), dtype)
    weights = actual.new_tensor([[0.1, 0.2], [0.3, 0.4]])
    (actual * weights).sum().backward()
    (expected * weights.double()).sum().backward()
    for leaf, exact in [(query, exact_query), (reference, exact_reference)]:
        assert leaf.grad is not None and exact.grad is not None
        assert leaf.grad.isfinite().all()
        assert_close(leaf.grad, exact.grad, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_snr_contrastive_training_keeps_positive_loss_and_gradients(dtype):
    x = rows(dtype).requires_grad_()
    exact = x.detach().double().requires_grad_()
    normalized = reference_rows(exact, dtype, True)
    pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
    expected = torch.stack(
        [reference_ratio(normalized[i], normalized[j]) for i, j in pairs]
    ).mean()
    actual = SignalToNoiseRatioContrastiveLoss()(
        x, torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    )
    assert actual > 0
    assert_close(actual, expected, dtype)
    actual.backward()
    expected.backward()
    assert x.grad is not None and exact.grad is not None
    assert x.grad.isfinite().all()
    assert x.grad.abs().max() > 0
    assert_close(x.grad, exact.grad, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_float32_variances_are_not_narrowed_by_autocast(dtype):
    x = rows(torch.float32)
    with torch.autocast(TEST_DEVICE.type, dtype=dtype):
        actual = SNRDistance()(x)
    assert actual.dtype == torch.float32
    assert_close(actual, SNRDistance()(x.double()), torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_empty_query_and_pairwise_shapes_keep_dtype(dtype):
    x = rows(dtype)
    distance = SNRDistance()
    assert distance(x[:0], x).shape == (0, 4)
    assert distance(x[:0], x).dtype == dtype
    assert distance.pairwise(x[:0], x[:0]).shape == (0,)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_truly_constant_rows_remain_undefined(dtype):
    x = torch.ones(2, 512, dtype=dtype, device=TEST_DEVICE)
    assert SNRDistance()(x).isnan().all()


def test_empty_miner_selection_remains_differentiable_zero():
    x = rows(torch.float16).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
    result = SignalToNoiseRatioContrastiveLoss()(
        x, labels, PairIndices(empty, empty, empty, empty)
    )
    result.backward()
    assert result.item() == 0
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) == 0
