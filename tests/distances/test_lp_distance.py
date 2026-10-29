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
