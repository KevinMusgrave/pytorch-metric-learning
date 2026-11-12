import math

import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import ManifoldLoss


def inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(4)
    embeddings = torch.randn(4, 3, generator=generator, dtype=torch.float64).to(dtype)
    proxies = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=dtype)
    return (
        embeddings.requires_grad_(),
        proxies.requires_grad_(),
        torch.tensor([0, 1, 0, 1]),
    )


def reference_loss(embeddings, proxies, labels, alpha, lambda_c, margin=5e-4):
    nodes = F.normalize(torch.cat((embeddings, proxies)), dim=1)
    identity = torch.eye(len(nodes), dtype=nodes.dtype)
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
    loss = ManifoldLoss(3, 2, alpha=alpha, lambda_c=lambda_c).double()
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
    loss = ManifoldLoss(3, 2, alpha=alpha).double()
    assert torch.autograd.gradcheck(
        lambda x, p: evaluate(loss, x, p, labels), (embeddings, proxies)
    )


@pytest.mark.parametrize("lambda_c", [0.0, 1.0, math.inf])
def test_valid_alpha_near_one_keeps_a_nonsingular_double_system(lambda_c):
    embeddings, proxies, labels = inputs()
    alpha = 0.99999999
    loss = ManifoldLoss(3, 2, alpha=alpha, lambda_c=lambda_c).double()
    actual = evaluate(loss, embeddings, proxies, labels)
    expected = reference_loss(embeddings, proxies, labels, alpha, lambda_c)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    gradients = torch.autograd.grad(actual, (embeddings, proxies))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_lower_precision_manifold_still_supports_inverse_and_backward(dtype):
    embeddings, proxies, labels = inputs(dtype)
    loss = ManifoldLoss(3, 2).to(dtype)
    actual = evaluate(loss, embeddings, proxies, labels)
    expected = reference_loss(embeddings.double(), proxies.double(), labels, 0.8, 1.0)
    torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=0.005)
    assert actual.dtype == dtype
    gradients = torch.autograd.grad(actual, (embeddings, proxies))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
