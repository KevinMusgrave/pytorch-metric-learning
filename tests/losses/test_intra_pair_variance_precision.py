import pytest
import torch

from pytorch_metric_learning import distances, losses, reducers

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("eps", [0.0, 0.001])
@pytest.mark.parametrize("wide_range", [False, True])
def test_intra_pair_variance_preserves_values_and_gradients_before_reduction(
    dtype, positive, inverted, eps, wide_range
):
    spacing = 0.00390625 if dtype == torch.bfloat16 else 0.00048828125
    values = [0.0, 1000.0] if wide_range else [0.5, 0.5 + spacing]
    matrix = torch.zeros(4, 4, device=TEST_DEVICE, dtype=dtype)
    matrix[0, 1], matrix[2, 3] = values
    matrix.requires_grad_()
    anchors = torch.tensor([0, 2], device=TEST_DEVICE)
    others = torch.tensor([1, 3], device=TEST_DEVICE)
    empty = torch.empty(0, device=TEST_DEVICE, dtype=torch.long)
    pairs = (
        (anchors, others, empty, empty) if positive else (empty, empty, anchors, others)
    )
    metric = (
        distances.DotProductSimilarity(normalize_embeddings=False)
        if inverted
        else distances.LpDistance(normalize_embeddings=False)
    )
    model = losses.IntraPairVarianceLoss(pos_eps=eps, neg_eps=eps, distance=metric)
    actual = model(
        torch.zeros(4, 2, device=TEST_DEVICE, dtype=dtype),
        indices_tuple=pairs,
        mat=matrix,
    )
    actual_gradient = torch.autograd.grad(actual, matrix)[0][anchors, others]
    reference = matrix.detach().double()[anchors, others].requires_grad_()
    mean = reference.sum() / len(reference)
    residual = (
        (1 - eps) * mean - reference
        if positive == inverted
        else reference - (1 + eps) * mean
    )
    expected = residual.clamp_min(0).square().sum() / len(reference)
    expected_gradient = torch.autograd.grad(expected, reference)[0]
    tolerance = 2e-4 if dtype != torch.float64 else 1e-12
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=1e-10)
    torch.testing.assert_close(
        actual_gradient, expected_gradient.to(dtype), rtol=2e-3, atol=1e-7
    )
    assert actual.dtype == (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_intra_pair_variance_keeps_small_positive_terms_for_nonzero_reduction(dtype):
    values = torch.tensor([0.5, 0.50390625], device=TEST_DEVICE, dtype=dtype)
    matrix = torch.zeros(4, 4, device=TEST_DEVICE, dtype=dtype)
    matrix[0, 1], matrix[2, 3] = values
    matrix.requires_grad_()
    empty = torch.empty(0, device=TEST_DEVICE, dtype=torch.long)
    pairs = (
        torch.tensor([0, 2], device=TEST_DEVICE),
        torch.tensor([1, 3], device=TEST_DEVICE),
        empty,
        empty,
    )
    model = losses.IntraPairVarianceLoss(
        pos_eps=0, reducer=reducers.AvgNonZeroReducer()
    )
    actual = model(
        torch.zeros(4, 2, device=TEST_DEVICE, dtype=dtype),
        indices_tuple=pairs,
        mat=matrix,
    )
    expected = ((values.double()[1] - values.double()[0]) / 2).square()
    torch.testing.assert_close(actual.double(), expected)
    gradient = torch.autograd.grad(actual, matrix)[0]
    assert gradient.abs().max() > 0


def test_intra_pair_variance_empty_selection_remains_differentiable_zero():
    query = torch.eye(3, device=TEST_DEVICE, requires_grad=True)
    empty = torch.empty(0, device=TEST_DEVICE, dtype=torch.long)
    actual = losses.IntraPairVarianceLoss()(query, indices_tuple=(empty,) * 4)
    actual.backward()
    assert actual.item() == 0
    torch.testing.assert_close(query.grad, torch.zeros_like(query))
