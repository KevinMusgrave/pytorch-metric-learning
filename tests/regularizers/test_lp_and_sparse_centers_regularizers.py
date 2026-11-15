import itertools
import math

import pytest
import torch

from pytorch_metric_learning.distances import DotProductSimilarity, LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import MeanReducer, SumReducer
from pytorch_metric_learning.regularizers import (
    LpRegularizer,
    SparseCentersRegularizer,
    ZeroMeanRegularizer,
)

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import assert_close, assert_matches_float64

ROWS = torch.tensor(
    [
        [0.5, -1.0, 2.0],
        [1.5, 0.75, -0.5],
        [-0.25, 0.5, 0.75],
        [2.0, -1.5, 0.25],
        [0.0, 1.0, -1.0],
        [-1.0, -0.5, 0.5],
    ],
    dtype=torch.float64,
)


def row_norms(x, p, power):
    return torch.stack([row.abs().pow(p).sum().pow(1 / p) ** power for row in x])


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize(("p", "power"), [(2, 1), (1, 1), (2, 2), (3, 0.5)])
def test_lp_regularizer_matches_mean_of_row_norms(p, power, dtype):
    regularizer = LpRegularizer(p=p, power=power)
    assert_matches_float64(
        regularizer, lambda x: row_norms(x, p, power).mean(), [ROWS], dtype
    )


@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_lp_regularizer_uses_its_reducer(dtype):
    regularizer = LpRegularizer(power=2, reducer=SumReducer())
    assert_matches_float64(
        regularizer, lambda x: row_norms(x, 2, 2).sum(), [ROWS], dtype
    )


def test_lp_regularizer_gradients_match_finite_differences():
    rows = ROWS.to(TEST_DEVICE).requires_grad_()
    for p, power in ((2, 1), (1, 2), (3, 0.5)):
        assert torch.autograd.gradcheck(LpRegularizer(p=p, power=power), (rows,))


def sparse_centers_reference(weight, num_classes, centers_per_class):
    centers = weight / torch.linalg.vector_norm(weight, dim=1, keepdim=True)
    total = weight.new_zeros(())
    for c in range(num_classes):
        rows = range(c * centers_per_class, (c + 1) * centers_per_class)
        for i, j in itertools.combinations(rows, 2):
            total = total + torch.sqrt(2 - 2 * (centers[i] @ centers[j]))
    return total / (num_classes * centers_per_class * (centers_per_class - 1))


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize(("num_classes", "centers_per_class"), [(3, 2), (2, 3)])
def test_sparse_centers_matches_softtriple_regularizer(
    num_classes, centers_per_class, dtype
):
    regularizer = SparseCentersRegularizer(num_classes, centers_per_class).to(
        TEST_DEVICE
    )
    assert_matches_float64(
        regularizer,
        lambda x: sparse_centers_reference(x, num_classes, centers_per_class),
        [ROWS],
        dtype,
    )


def test_sparse_centers_is_zero_when_each_class_merges():
    weight = ROWS.clone().to(TEST_DEVICE)
    weight[1] = weight[0] * 2
    weight[3] = weight[2]
    weight[5] = weight[4] * 0.5
    assert SparseCentersRegularizer(3, 2)(weight).item() == pytest.approx(0, abs=1e-6)


def test_sparse_centers_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SparseCentersRegularizer(3, 1)
    with pytest.raises(TypeError):
        SparseCentersRegularizer(3, 2, distance=DotProductSimilarity())
    with pytest.raises(ValueError):
        SparseCentersRegularizer(3, 2, reducer=MeanReducer())(ROWS)


@pytest.mark.parametrize("regularizer_class", [LpRegularizer, ZeroMeanRegularizer])
def test_distance_free_regularizers_reject_distance(regularizer_class):
    options: dict = {"distance": LpDistance()}
    with pytest.raises(TypeError):
        regularizer_class(**options)
    rows = ROWS.to(TEST_DEVICE)
    summed = regularizer_class(reducer=SumReducer())(rows)
    averaged = regularizer_class()(rows)
    torch.testing.assert_close(summed, averaged * len(rows))


def unbounded_regularizer(kind, reducer=None):
    if kind == "squared_l2":
        return LpRegularizer(power=2, reducer=reducer)
    if kind == "l1":
        return LpRegularizer(p=1, reducer=reducer)
    return ZeroMeanRegularizer(reducer=reducer)


def unbounded_reference(x, kind, average):
    rows = x.detach().double().cpu().tolist()
    values = []
    gradients = []
    for row in rows:
        if kind == "squared_l2":
            values.append(math.fsum(value * value for value in row))
            gradients.append([2 * value for value in row])
        elif kind == "l1":
            values.append(math.fsum(abs(value) for value in row))
            gradients.append([(value > 0) - (value < 0) for value in row])
        else:
            total = math.fsum(row)
            values.append(abs(total))
            gradients.append([(total > 0) - (total < 0)] * len(row))
    divisor = len(rows) if average else 1
    gradient = torch.tensor(gradients, dtype=torch.float64, device=x.device)
    return math.fsum(values) / divisor, gradient / divisor


@pytest.mark.parametrize("kind", ["squared_l2", "l1", "zero_mean"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("average", [False, True])
def test_unbounded_regularizers_keep_large_terms_until_reduction(kind, dtype, average):
    first = [300.0, 0.0] if kind == "squared_l2" else [40000.0, 40000.0]
    x = torch.tensor(
        [first, [3.0, -1.0], [0.0, 0.0]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    reducer = MeanReducer() if average else SumReducer()
    actual = unbounded_regularizer(kind, reducer)(x)
    value, gradient = unbounded_reference(x, kind, average)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert actual.isfinite()
    assert_close(actual, value, expected_dtype)
    assert actual.dtype == expected_dtype
    actual.backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert_close(x.grad, gradient, dtype)


@pytest.mark.parametrize("kind", ["squared_l2", "l1", "zero_mean"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_regularizer_weight_keeps_representable_loss_and_gradient(kind, dtype):
    x = torch.tensor(
        [[40000.0, 40000.0]], dtype=dtype, device=TEST_DEVICE, requires_grad=True
    )
    weight = 1e-5 if kind == "squared_l2" else 0.25
    loss = ContrastiveLoss(
        embedding_regularizer=unbounded_regularizer(kind), embedding_reg_weight=weight
    )
    actual = loss(x, torch.tensor([0], device=TEST_DEVICE))
    value, gradient = unbounded_reference(x, kind, True)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert actual.isfinite()
    assert_close(actual, value * weight, expected_dtype)
    actual.backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert_close(x.grad, gradient * weight, dtype)


@pytest.mark.parametrize("kind", ["squared_l2", "l1", "zero_mean"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [0, 2])
def test_unbounded_regularizers_keep_empty_and_zero_gradients(kind, dtype, size):
    x = torch.zeros(size, 3, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    actual = unbounded_regularizer(kind)(x)
    assert actual.item() == 0
    if size == 0:
        assert actual.dtype == dtype
    actual.backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize("kind", ["squared_l2", "l1", "zero_mean"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_regularizer_statistics_stay_float32_under_autocast(kind, dtype):
    x = torch.tensor(
        [[300.0, -300.0], [3.0, -1.0]],
        dtype=torch.float32,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    value, gradient = unbounded_reference(x, kind, True)
    with torch.autocast(TEST_DEVICE.type, dtype=dtype):
        actual = unbounded_regularizer(kind)(x)
    assert actual.dtype == torch.float32
    assert_close(actual, value, torch.float32)
    actual.backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert_close(x.grad, gradient, torch.float32)
