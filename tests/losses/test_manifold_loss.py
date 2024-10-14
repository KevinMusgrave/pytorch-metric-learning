import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from pytorch_metric_learning.losses import ManifoldLoss

from .. import TEST_DEVICE, TEST_DTYPES


def pairwise_similarity(x, y=None):
    # x_norm = (x**2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
    else:
        y_t = torch.transpose(x, 0, 1)

    dist = torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# DIRECTLY ASKED Nicolas Aziere.
# This code is copied from the official implementation
# so that we can make sure our implementation returns the same result.
# Some minor changes were made to avoid errors during testing.
# Every change in the original code is reported and explained.
class OriginalImplementationManifoldLoss(Module):
    def __init__(self, proxies, alpha, lambdaC=1.0, distance=F.cosine_similarity):
        """
        proxies : P x D , proxy embeddings (one proxy per class randomly initialized, instance of nn.Parameter)
        alpha : float, random walk parameter
        lambdaC : float, regularization weight
        distance : func, distance function to use
        """
        super(OriginalImplementationManifoldLoss, self).__init__()
        self.alpha = alpha
        self.lambdaC = lambdaC
        self.proxy = proxies / proxies.norm(p=2)  # Removed .cuda() to avoid errors
        self.nb_proxy = proxies.size(0)
        self.d = distance

    def get_Matrix(self, x):
        """
        x : B x D , feature embeddings
        return
            A: B x B the approximated rank matrix
        """
        # Removed x = x.cuda() to avoid errors

        # Construct Affinity matrix
        W = pairwise_similarity(x)
        # Gaussian kernel
        W = torch.exp(
            W / 0.5
        )  # Increased variance from 0.05 to 0.5 in order to account for dtype Half in our code

        Y = torch.eye(
            len(W), dtype=W.dtype, device=x.device
        )  # Added dtype and device cast to avoid errors

        ## Set diagonal to 0 ?????
        W = W - W * Y
        D = torch.diag(torch.pow(torch.sum(W, dim=1), -0.5))
        D[D == float("Inf")] = 0.0
        S = torch.mm(torch.mm(D, W), D)  # Removed .cuda() to avoid errors

        # Solve random walk closed form
        dt = S.dtype
        L = torch.inverse(
            Y.float() - self.alpha * S.float()
        )  # Added dtype cast to avoid errors
        L = L.to(dt)
        A = (1 - self.alpha) * torch.mm(L, Y)
        return A

    def forward(self, fvec, fLvec, fvecs_add=None):
        """
        fvec : B1 x D , current batch of feature embedding
        fLvec : B1 , current batch of GT labels
        fvecs_add : B2 x D , batch of additionnal contextual features to fill the manifold
        """
        fLvec = fLvec.tolist()
        N = len(fLvec)

        if fvecs_add is not None:
            fvec = torch.cat((fvec, self.proxy, fvecs_add), 0)
        else:
            fvec = torch.cat((fvec, self.proxy), 0)
        # normalization (as in the original proxy method)
        fvec = fvec / fvec.norm(p=2, dim=1).view(-1, 1)

        # Rank matrix
        A = self.get_Matrix(fvec)
        A_p = A[N : N + self.nb_proxy].clone()
        A = A[:N]
        loss_intrinsic = torch.zeros(
            1, dtype=fvec.dtype, device=fvec.device
        )  # Modified to avoid errors
        loss_context = torch.zeros(
            1, dtype=fvec.dtype, device=fvec.device
        )  # Modified to avoid errors

        for i in range(N):
            loss_neg1_intrinsic = torch.zeros(
                1, dtype=fvec.dtype, device=fvec.device
            )  # Removed .cuda() to avoid errors                                   # NoQA
            loss_neg1_context = torch.zeros(
                1, dtype=fvec.dtype, device=fvec.device
            )  # Removed .cuda() to avoid errors                                   # NoQA
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

            loss_intrinsic += torch.log(
                1.0 + loss_neg1_intrinsic
            )  # + torch.log(torch.add(loss_neg2_intrinsec, 1.0))
            loss_context += torch.log(
                1.0 + loss_neg1_context
            )  # + torch.log(torch.add(loss_neg2_context, 1.0))

        loss_intrinsic /= N
        loss_context /= N

        return loss_intrinsic + self.lambdaC * loss_context


def loss_incorrect_descriptors_dim():
    embeddings = torch.randn(10, 10)
    loss_fn = ManifoldLoss(l=999)
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

            self.assertRaises(
                ValueError, lambda: ManifoldLoss(l=embedding_size, lambdaC=-999)
            )
            self.assertRaises(AssertionError, loss_incorrect_descriptors_dim)

            loss_func = ManifoldLoss(
                l=embedding_size, K=n_proxies, alpha=alpha, lambdaC=0, margin=0.0
            )
            loss_func.proxies.data = (
                proxies.data
            )  # In order to have same initializations
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
                l=embedding_size, K=n_proxies, alpha=alpha, margin=0.0
            )  # Original implementation does
            # not consider margin                   # NoQA

            loss_func.proxies.data = (
                proxies.data
            )  # In order to have same initializations
            loss = loss_func(embeddings, indices_tuple=labels)
            loss.backward()

            rtol = 1e-2 if dtype == torch.float16 else 1e-5

            self.assertTrue(torch.isclose(original_loss, loss, rtol=rtol))


if __name__ == "__main__":
    unittest.main()
