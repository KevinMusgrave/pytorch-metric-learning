import pytest
import torch

from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.regularizers import SphericalEmbeddingRegularizer

from .. import TEST_DEVICE, TEST_DTYPES


@pytest.mark.parametrize("dtype", [*TEST_DTYPES, torch.bfloat16])
@pytest.mark.parametrize("sum_reduction", [False, True])
@pytest.mark.parametrize("radii,weight", [([1.0, 1001.0], 1e-4), ([1e-4, 2e-4], 1e4)])
def test_sec_retains_precision_for_later_weighting(dtype, sum_reduction, radii, weight):
    x = torch.tensor([[r, 0] for r in radii], dtype=dtype, device=TEST_DEVICE)
    x.requires_grad_()
    reducer = reducers.SumReducer() if sum_reduction else reducers.MeanReducer()
    value = SphericalEmbeddingRegularizer(reducer=reducer)(x)
    gap = x.detach().double()[1, 0] - x.detach().double()[0, 0]
    factor = 2 if sum_reduction else 1
    expected = gap.square() * (factor * weight / 4)
    expected_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    assert value.dtype == expected_dtype
    weighted = value * weight
    torch.testing.assert_close(weighted, expected.to(expected_dtype), rtol=2e-6, atol=0)
    weighted.backward()
    gradient = torch.zeros_like(x)
    gradient[0, 0] = -gap * (factor * weight / 2)
    gradient[1, 0] = gap * (factor * weight / 2)
    torch.testing.assert_close(x.grad, gradient)


@pytest.mark.parametrize("dtype", [*TEST_DTYPES, torch.bfloat16])
def test_sec_composes_with_a_finite_weighted_training_loss(dtype):
    x = torch.tensor(
        [[1.0, 0.0], [1001.0, 0.0], [0.0, 1.0], [0.0, 1001.0]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    regularized = losses.ContrastiveLoss(
        embedding_regularizer=SphericalEmbeddingRegularizer(),
        embedding_reg_weight=1e-4,
    )
    result = regularized(x, labels)
    rows = x.detach().double()
    gap = rows[1, 0] - rows[0, 0]
    expected = gap.square() * 1e-4 / 4
    torch.testing.assert_close(result.double(), expected, rtol=2e-6, atol=0)
    result.backward()
    gradient = torch.zeros_like(x)
    gradient[0, 0] = gradient[2, 1] = -gap * 1e-4 / 4
    gradient[1, 0] = gradient[3, 1] = gap * 1e-4 / 4
    torch.testing.assert_close(x.grad, gradient)
