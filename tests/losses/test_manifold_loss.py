import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import Module

from pytorch_metric_learning.distances import DotProductSimilarity, LpDistance
from pytorch_metric_learning.losses import ManifoldLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers

KERNEL_TEMPERATURE_MATCHING_LIBRARY = 0.5
ORIGINAL_IMPLEMENTATION_MARGIN = 0.0


def pairwise_similarity(x, y=None):
    y_t = torch.transpose(y if y is not None else x, 0, 1)
    dist = torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


class OriginalImplementationManifoldLoss(Module):
    def __init__(self, proxies, alpha, lambdaC=1.0, distance=F.cosine_similarity):
        super().__init__()
        self.alpha = alpha
        self.lambdaC = lambdaC
        self.proxy = proxies / proxies.norm(p=2)
        self.nb_proxy = proxies.size(0)
        self.d = distance

    def get_Matrix(self, x):
        W = pairwise_similarity(x)
        W = torch.exp(W / KERNEL_TEMPERATURE_MATCHING_LIBRARY)

        Y = torch.eye(len(W), dtype=W.dtype, device=x.device)

        W = W - W * Y
        D = torch.diag(torch.pow(torch.sum(W, dim=1), -0.5))
        D[float("Inf") == D] = 0.0
        S = torch.mm(torch.mm(D, W), D)

        dt = S.dtype
        L = torch.inverse(Y.float() - self.alpha * S.float())
        L = L.to(dt)
        A = (1 - self.alpha) * torch.mm(L, Y)
        return A

    def forward(self, fvec, fLvec, fvecs_add=None):
        fLvec = fLvec.tolist()
        N = len(fLvec)

        if fvecs_add is not None:
            fvec = torch.cat((fvec, self.proxy, fvecs_add), 0)
        else:
            fvec = torch.cat((fvec, self.proxy), 0)
        fvec = fvec / fvec.norm(p=2, dim=1).view(-1, 1)

        A = self.get_Matrix(fvec)
        A_p = A[N : N + self.nb_proxy].clone()
        A = A[:N]
        loss_intrinsic = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
        loss_context = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)

        for i in range(N):
            loss_neg1_intrinsic = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
            loss_neg1_context = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
            dist_pos = self.d(
                torch.unsqueeze(A[i], 0), torch.unsqueeze(A_p[fLvec[i]], 0)
            )

            for j in range(self.nb_proxy):
                if fLvec[i] != j:
                    val1_context = (
                        self.d(torch.unsqueeze(A[i], 0), torch.unsqueeze(A_p[j], 0))
                        - dist_pos
                    )
                    val1_intrinsic = A[i, N + j] - A[i, N + fLvec[i]]
                    if val1_context > 0:
                        loss_neg1_context += torch.exp(val1_context)
                    if val1_intrinsic > 0:
                        loss_neg1_intrinsic += torch.exp(val1_intrinsic)

            loss_intrinsic += torch.log(1.0 + loss_neg1_intrinsic)
            loss_context += torch.log(1.0 + loss_neg1_context)

        loss_intrinsic /= N
        loss_context /= N

        return loss_intrinsic + self.lambdaC * loss_context


def loss_incorrect_descriptors_dim():
    embeddings = torch.randn(10, 10)
    loss_fn = ManifoldLoss(embedding_size=999)
    loss_fn(embeddings)


class TestManifoldLoss(unittest.TestCase):
    def test_intrinsic_and_context_losses(self):
        torch.manual_seed(24)
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            batch_size, embedding_size = 32, 128
            n_proxies = 3

            embeddings = torch.randn(
                batch_size,
                embedding_size,
                device=TEST_DEVICE,
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.randint(0, n_proxies, size=(batch_size,), device=TEST_DEVICE)
            proxies = nn.Parameter(
                torch.randn(n_proxies, embedding_size, device=TEST_DEVICE, dtype=dtype)
            )
            alpha = 0.99

            original_loss_func = OriginalImplementationManifoldLoss(
                proxies, alpha=alpha, lambdaC=0
            )
            original_loss = original_loss_func(embeddings, labels)

            with self.assertRaises(ValueError):
                ManifoldLoss(embedding_size=embedding_size, lambda_c=-999)
            self.assertRaises(ValueError, loss_incorrect_descriptors_dim)

            loss_func = ManifoldLoss(
                embedding_size=embedding_size,
                num_proxies=n_proxies,
                alpha=alpha,
                lambda_c=0,
                margin=ORIGINAL_IMPLEMENTATION_MARGIN,
            )
            loss_func.to(device=proxies.device, dtype=proxies.dtype)
            with torch.no_grad():
                loss_func.proxies.copy_(proxies)
            loss = loss_func(embeddings, labels)
            loss.backward()

            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(original_loss, loss, rtol=rtol))

    def test_with_original_implementation(self):
        torch.manual_seed(24)
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            batch_size, embedding_size = 32, 128
            n_proxies = 5

            embeddings = torch.randn(
                batch_size,
                embedding_size,
                device=TEST_DEVICE,
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.randint(0, n_proxies, size=(batch_size,), device=TEST_DEVICE)

            proxies = nn.Parameter(
                torch.randn(n_proxies, embedding_size, device=TEST_DEVICE, dtype=dtype)
            )
            alpha = 0.99
            original_loss_func = OriginalImplementationManifoldLoss(
                proxies, alpha=alpha
            )
            original_loss = original_loss_func(embeddings, labels)

            loss_func = ManifoldLoss(
                embedding_size=embedding_size,
                num_proxies=n_proxies,
                alpha=alpha,
                margin=ORIGINAL_IMPLEMENTATION_MARGIN,
            )

            loss_func.to(device=proxies.device, dtype=proxies.dtype)
            with torch.no_grad():
                loss_func.proxies.copy_(proxies)
            loss = loss_func(embeddings, labels)
            loss.backward()

            rtol = helpers.rtol(dtype)

            self.assertTrue(torch.isclose(original_loss, loss, rtol=rtol))


if __name__ == "__main__":
    unittest.main()


def direct_dot_product_loss(embeddings, proxies, labels, alpha, margin):
    nodes = torch.cat((embeddings, proxies))
    N, identity = len(embeddings), torch.eye(len(nodes), dtype=nodes.dtype).to(nodes)
    W = torch.exp((nodes @ nodes.T).clamp_min(0) / 0.5) * (1 - identity)
    scale = W.sum(dim=1) ** -0.5
    S = scale[:, None] * W * scale[None, :]
    manifold = (1 - alpha) * torch.inverse(identity - alpha * S)
    context = F.cosine_similarity(manifold[:N, None, :], manifold[None, N:, :], dim=-1)
    total = torch.zeros((), dtype=nodes.dtype, device=nodes.device)
    for scores in (manifold[:N, N:], context):
        for row, label in enumerate(labels.tolist()):
            exponentials = [
                torch.exp(scores[row, j] - scores[row, label] + margin)
                for j in range(len(proxies))
                if j != label and scores[row, j] - scores[row, label] + margin >= 0
            ]
            total = (
                total + torch.log(1 + sum(exponentials, torch.zeros_like(total))) / N
            )
    return total


class TestManifoldLossScale(unittest.TestCase):
    def test_similarities_past_the_float32_exponential_range(self):
        torch.manual_seed(5)
        wide = (torch.randn(12, 8, device=TEST_DEVICE, dtype=torch.float64) * 4).round()
        labels = torch.arange(12, device=TEST_DEVICE) % 3
        proxies = nn.Parameter(
            torch.randn(3, 8, device=TEST_DEVICE, dtype=torch.float64) * 4
        )
        scores = wide @ wide.T
        self.assertGreater(scores.max().item() / 0.5, 89)
        self.assertLess(scores.max().item() / 0.5, 700)
        reference_input = wide.clone().requires_grad_()
        reference = direct_dot_product_loss(
            reference_input, proxies.detach(), labels, 0.8, 5e-4
        )
        reference.backward()
        assert reference_input.grad is not None
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                embeddings = wide.to(dtype).requires_grad_()
                loss_func = ManifoldLoss(8, 3, distance=DotProductSimilarity())
                loss_func.to(TEST_DEVICE, dtype)
                with torch.no_grad():
                    loss_func.proxies.copy_(proxies.to(dtype))
                loss = loss_func(embeddings, labels)
                loss.backward()
                assert embeddings.grad is not None
                torch.testing.assert_close(
                    loss.double(), reference, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    embeddings.grad.double(),
                    reference_input.grad,
                    rtol=2e-3,
                    atol=2e-3,
                )

    def test_similarities_past_the_float64_exponential_range(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(
                    [[-150.0, 0.0], [150.0, 0.0], [-150.0, 225.0], [150.0, 225.0]],
                    device=TEST_DEVICE,
                    dtype=dtype,
                    requires_grad=True,
                )
                labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
                loss_func = ManifoldLoss(2, 2, distance=DotProductSimilarity())
                loss = loss_func.to(TEST_DEVICE, dtype)(embeddings, labels)
                loss.backward()
                assert embeddings.grad is not None
                self.assertTrue(loss.isfinite())
                self.assertTrue(embeddings.grad.isfinite().all())

    def test_distance_must_be_a_similarity(self):
        with self.assertRaisesRegex(TypeError, "similarity"):
            ManifoldLoss(2, 2, distance=LpDistance(normalize_embeddings=False))


class TestManifoldLossOptions(unittest.TestCase):
    def test_infinite_lambda_uses_only_the_context_term(self):
        embeddings = torch.randn(12, 4, device=TEST_DEVICE)
        labels = torch.arange(3, device=TEST_DEVICE).repeat(4)
        torch.manual_seed(1)
        context_only = ManifoldLoss(4, num_proxies=3, lambda_c=math.inf).to(TEST_DEVICE)
        torch.manual_seed(1)
        weighted = ManifoldLoss(4, num_proxies=3, lambda_c=1.0).to(TEST_DEVICE)
        torch.manual_seed(1)
        intrinsic = ManifoldLoss(4, num_proxies=3, lambda_c=0.0).to(TEST_DEVICE)
        value = context_only(embeddings, labels)
        self.assertTrue(torch.isfinite(value))
        torch.testing.assert_close(
            value, weighted(embeddings, labels) - intrinsic(embeddings, labels)
        )

    def test_small_unlabeled_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_proxies"):
            ManifoldLoss(4, num_proxies=8).to(TEST_DEVICE)(
                torch.randn(4, 4, device=TEST_DEVICE)
            )

    def test_empty_labeled_batch_gives_differentiable_zero(self):
        loss_func = ManifoldLoss(4, num_proxies=3).to(TEST_DEVICE)
        embeddings = torch.randn(0, 4, device=TEST_DEVICE, requires_grad=True)
        labels = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
        loss = loss_func(embeddings, labels)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))
