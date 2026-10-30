import itertools

import pytest
import torch

from pytorch_metric_learning.distances import DotProductSimilarity, LpDistance
from pytorch_metric_learning.reducers import MeanReducer, SumReducer
from pytorch_metric_learning.regularizers import (
    LpRegularizer,
    SparseCentersRegularizer,
    ZeroMeanRegularizer,
)

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import assert_matches_float64

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
