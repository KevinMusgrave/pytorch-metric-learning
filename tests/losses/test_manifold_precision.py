import math

import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import ManifoldLoss

from .. import TEST_DEVICE


def inputs(dtype=torch.float64):
    generator = torch.Generator(device=TEST_DEVICE).manual_seed(4)
    embeddings = torch.randn(
        4, 3, generator=generator, dtype=torch.float64, device=TEST_DEVICE
    ).to(dtype)
    proxies = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=dtype, device=TEST_DEVICE
    )
    return (
        embeddings.requires_grad_(),
        proxies.requires_grad_(),
        torch.tensor([0, 1, 0, 1], device=TEST_DEVICE),
    )


def reference_loss(embeddings, proxies, labels, alpha, lambda_c, margin=5e-4):
    nodes = F.normalize(torch.cat((embeddings, proxies)), dim=1)
    identity = torch.eye(len(nodes), dtype=nodes.dtype, device=TEST_DEVICE)
    graph = (nodes @ nodes.T).clamp_min(0).div(0.5).exp() * (1 - identity)
    inverse_degree = graph.sum(dim=1).rsqrt()
    normalized = inverse_degree[:, None] * graph * inverse_degree[None, :]
    affinity = (1 - alpha) * torch.linalg.solve(identity - alpha * normalized, identity)
    n = len(embeddings)
    context = F.cosine_similarity(affinity[:n, None], affinity[None, n:], dim=-1)

    def ranking(scores):
        terms = []
        for i, label in enumerate(labels.tolist()):
            violations = torch.stack(
                [
                    scores[i, j] - scores[i, label] + margin
                    for j in range(len(proxies))
                    if j != label
                ]
            )
            terms.append(
                torch.log1p(torch.where(violations >= 0, violations.exp(), 0).sum())
            )
        return torch.stack(terms).mean()

    context_loss = ranking(context)
    if lambda_c == math.inf:
        return context_loss
    return ranking(affinity[:n, n:]) + lambda_c * context_loss


def evaluate(loss, embeddings, proxies, labels):
    return torch.func.functional_call(loss, {"proxies": proxies}, (embeddings, labels))


@pytest.mark.parametrize("alpha", [0.2, 0.8, 0.99])
@pytest.mark.parametrize("lambda_c", [0.0, 1.0, math.inf])
def test_float64_manifold_values_and_gradients(alpha, lambda_c):
    embeddings, proxies, labels = inputs()
    loss = ManifoldLoss(3, 2, alpha=alpha, lambda_c=lambda_c).to(TEST_DEVICE).double()
    actual = evaluate(loss, embeddings, proxies, labels)
    expected = reference_loss(embeddings, proxies, labels, alpha, lambda_c)
    actual_gradients = torch.autograd.grad(actual, (embeddings, proxies))
    expected_gradients = torch.autograd.grad(expected, (embeddings, proxies))
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-9, atol=1e-11
        )


@pytest.mark.parametrize("alpha", [0.8, 0.99])
def test_float64_manifold_passes_numerical_gradient_check(alpha):
    embeddings, proxies, labels = inputs()
    loss = ManifoldLoss(3, 2, alpha=alpha).to(TEST_DEVICE).double()
    assert torch.autograd.gradcheck(
        lambda x, p: evaluate(loss, x, p, labels), (embeddings, proxies)
    )


@pytest.mark.parametrize("lambda_c", [0.0, 1.0, math.inf])
def test_valid_alpha_near_one_keeps_a_nonsingular_double_system(lambda_c):
    embeddings, proxies, labels = inputs()
    alpha = 0.99999999
    loss = ManifoldLoss(3, 2, alpha=alpha, lambda_c=lambda_c).to(TEST_DEVICE).double()
    actual = evaluate(loss, embeddings, proxies, labels)
    expected = reference_loss(embeddings, proxies, labels, alpha, lambda_c)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    gradients = torch.autograd.grad(actual, (embeddings, proxies))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_lower_precision_manifold_still_supports_inverse_and_backward(dtype):
    embeddings, proxies, labels = inputs(dtype)
    loss = ManifoldLoss(3, 2).to(TEST_DEVICE).to(dtype)
    actual = evaluate(loss, embeddings, proxies, labels)
    expected = reference_loss(embeddings.double(), proxies.double(), labels, 0.8, 1.0)
    torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=0.005)
    assert actual.dtype == dtype
    gradients = torch.autograd.grad(actual, (embeddings, proxies))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("dtype", "margin"),
    [
        (torch.float16, 12.0),
        (torch.bfloat16, 90.0),
        (torch.float32, 90.0),
        (torch.float64, 90.0),
    ],
)
@pytest.mark.parametrize("lambda_c", [0.0, 1.0, math.inf])
def test_large_manifold_margin_has_finite_values_and_gradients(dtype, margin, lambda_c):
    embeddings, proxies, labels = inputs(dtype)
    loss = ManifoldLoss(3, 2, alpha=0.8, lambda_c=lambda_c, margin=margin).to(
        TEST_DEVICE, dtype
    )
    actual = evaluate(loss, embeddings, proxies, labels)
    expected = reference_loss(
        embeddings.double(), proxies.double(), labels, 0.8, lambda_c, margin
    )
    actual_gradients = torch.autograd.grad(
        actual, (embeddings, proxies), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, (embeddings, proxies))
    assert torch.isfinite(actual)
    assert actual.dtype == dtype
    tolerance = 0.05 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    absolute = 0.003 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(
        actual.double(), expected, rtol=tolerance, atol=tolerance
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=tolerance, atol=absolute
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ranking_past_double_exponential_range_has_closed_form_gradient(dtype):
    scores = torch.zeros((2, 4), dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    rows = torch.arange(2, device=TEST_DEVICE)
    labels = torch.tensor([0, 2], device=TEST_DEVICE)
    loss = ManifoldLoss(3, 4, margin=750.0).to(TEST_DEVICE, dtype)
    actual = loss.ranking_term(scores, rows, labels)
    expected = torch.tensor(750.0 + math.log(3), dtype=dtype, device=TEST_DEVICE)
    gradient = torch.autograd.grad(actual, scores)[0]
    expected_gradient = torch.full_like(scores, 1 / 6)
    expected_gradient[rows, labels] = -0.5
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient, expected_gradient, rtol=5e-5, atol=1e-6)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_single_proxy_keeps_zero_ranking_and_gradients(dtype):
    embeddings, _, labels = inputs(dtype)
    labels = torch.zeros_like(labels)
    loss = ManifoldLoss(3, 1, margin=90.0).to(TEST_DEVICE, dtype)
    actual = loss(embeddings, labels)
    gradients = torch.autograd.grad(actual, (embeddings, loss.proxies))
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
