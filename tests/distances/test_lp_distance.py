import pytest
import torch

from pytorch_metric_learning.distances import LpDistance

from .. import TEST_DEVICE


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_half_precision_values_and_gradients(dtype, p, power, normalize):
    query = torch.tensor(
        [[0.25, -0.5, 1.0], [1.5, 0.75, -1.0]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    reference = torch.tensor(
        [[-1.0, 1.25, 0.5], [0.75, -1.5, 2.0], [-0.5, -0.25, -1.25]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    expected_query = query.detach().double().requires_grad_()
    expected_reference = reference.detach().double().requires_grad_()
    q, r = expected_query, expected_reference
    if normalize:
        q = q / torch.linalg.vector_norm(q, ord=p, dim=1, keepdim=True)
        r = r / torch.linalg.vector_norm(r, ord=p, dim=1, keepdim=True)
    expected = torch.linalg.vector_norm(q[:, None] - r[None, :], ord=p, dim=2)
    expected = expected**power

    distance = LpDistance(normalize_embeddings=normalize, p=p, power=power)
    actual = distance(query, reference)
    assert actual.dtype == dtype
    assert actual.device == query.device
    weights = query.new_tensor([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]])
    actual_gradients = torch.autograd.grad(
        (actual * weights).mean(), (query, reference)
    )
    expected_gradients = torch.autograd.grad(
        (expected * weights.double()).mean(), (expected_query, expected_reference)
    )
    tolerance = torch.finfo(dtype).eps
    for result, target in zip(
        (actual, *actual_gradients), (expected, *expected_gradients), strict=True
    ):
        torch.testing.assert_close(
            result.double(), target, rtol=4 * tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [24, 32])
def test_half_precision_nearby_rows(dtype, batch):
    query = torch.ones(batch, 128, dtype=dtype, device=TEST_DEVICE)
    reference = query.clone()
    reference[:, 0] = 1 - torch.finfo(dtype).eps
    query.requires_grad_()
    reference.requires_grad_()
    actual = LpDistance(normalize_embeddings=False)(query, reference)
    expected = torch.linalg.vector_norm(
        query.detach().double()[:, None] - reference.detach().double()[None, :], dim=2
    )
    gradient = torch.autograd.grad(actual.sum(), query)[0]
    tolerance = torch.finfo(dtype).eps
    assert torch.all(expected > 0)
    torch.testing.assert_close(actual.double(), expected, rtol=4 * tolerance, atol=0)
    torch.testing.assert_close(
        gradient.double()[:, 0], torch.full_like(expected[:, 0], float(batch))
    )
    assert torch.all(gradient[:, 1:] == 0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_pairwise_equals_matrix_diagonal(dtype, p, normalize):
    query = torch.tensor(
        [[0.25, -0.5, 1.0], [1.5, 0.75, -1.0], [0.5, 0.5, 0.5]],
        dtype=dtype,
        device=TEST_DEVICE,
    )
    reference = query.clone()
    reference[1] = torch.tensor([-1.0, 1.25, 0.5], dtype=dtype)
    distance = LpDistance(normalize_embeddings=normalize, p=p)
    pairwise = distance.pairwise(query, reference)
    assert pairwise.dtype == dtype
    diagonal = distance(query, reference).diagonal()
    assert pairwise[0] == diagonal[0] == 0 and pairwise[2] == diagonal[2] == 0
    q, r = query.double(), reference.double()
    if normalize:
        q = q / torch.linalg.vector_norm(q, ord=p, dim=1, keepdim=True)
        r = r / torch.linalg.vector_norm(r, ord=p, dim=1, keepdim=True)
    expected = torch.linalg.vector_norm(q - r, ord=p, dim=1)
    tolerance = torch.finfo(dtype).eps
    for actual in (pairwise, diagonal):
        torch.testing.assert_close(
            actual.double(), expected, rtol=4 * tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("p", [1, 2, 3])
def test_pairwise_gradients_at_identical_rows_match_matrix_path(dtype, p):
    rows = torch.tensor(
        [[0.25, -0.5, 1.0], [1.5, 0.75, -1.0]], dtype=dtype, device=TEST_DEVICE
    )
    distance = LpDistance(normalize_embeddings=False, p=p)
    gradients = []
    for compute in (
        lambda x, y: distance.pairwise(x, y),
        lambda x, y: distance(x, y).diagonal(),
    ):
        query = rows.clone().requires_grad_()
        reference = rows.clone().requires_grad_()
        gradients.append(
            torch.autograd.grad(compute(query, reference).sum(), (query, reference))
        )
    for pairwise_gradient, matrix_gradient in zip(*gradients, strict=True):
        assert torch.isfinite(pairwise_gradient).all()
        assert torch.equal(pairwise_gradient, torch.zeros_like(pairwise_gradient))
        assert torch.equal(pairwise_gradient, matrix_gradient)
