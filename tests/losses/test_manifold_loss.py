import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import Module

from pytorch_metric_learning.losses import ManifoldLoss

from .. import TEST_DEVICE, TEST_DTYPES

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
            self.assertRaises(AssertionError, loss_incorrect_descriptors_dim)

            loss_func = ManifoldLoss(
                embedding_size=embedding_size,
                num_proxies=n_proxies,
                alpha=alpha,
                lambda_c=0,
                margin=ORIGINAL_IMPLEMENTATION_MARGIN,
            )
            loss_func.proxies.data = proxies.data
            loss = loss_func(embeddings, indices_tuple=labels)
            loss.backward()

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
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

            loss_func.proxies.data = proxies.data
            loss = loss_func(embeddings, indices_tuple=labels)
            loss.backward()

            rtol = 1e-2 if dtype == torch.float16 else 1e-5

            self.assertTrue(torch.isclose(original_loss, loss, rtol=rtol))


if __name__ == "__main__":
    unittest.main()
