import unittest

import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_metric_learning.losses import ManifoldLoss
from tests import TEST_DTYPES, TEST_DEVICE
from tests.zzz_testing_utils.testing_utils import angle_to_coord


def pairwise_similarity(x, y=None):
    # x_norm = (x**2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        # y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        # y_norm = x_norm.view(1, -1)

    dist = torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
    # return dist


######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# DIRECTLY ASKED TO Nicolas Aziere
# This code is copied from the official implementation
# so that we can make sure our implementation returns the same result.
# Some minor changes were made to avoid errors during testing.
# Every modification is reported.
# It's copied under the MIT license.
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
        self.proxy = proxies.cuda() / proxies.norm(p=2)
        self.nb_proxy = proxies.size(0)
        self.d = distance

    def get_Matrix(self, x):
        """
        x : B x D , feature embeddings
        return
            A: B x B the approximated rank matrix
        """
        x = x.cuda()

        # Construct Affinity matrix
        # W = pairwise_distances(x)
        W = pairwise_similarity(x)
        # Gaussian kernel
        W = torch.exp(W / 0.5)  # Increased variance from 0.05 to 0.5 in order to account for dtype Half in our code

        Y = torch.eye(len(W), dtype=W.dtype).cuda()  # Added dtype cast to avoid errors

        ## Set diagonal to 0 ?????
        W = W - W * Y
        # print(W[0])
        D = torch.diag(torch.pow(torch.sum(W, dim=1), -0.5))
        D[D == float("Inf")] = 0.0
        S = torch.mm(torch.mm(D, W), D).cuda()

        # Solve random walk closed form
        dt = S.dtype
        L = torch.inverse(Y.float() - self.alpha * S.float())  # Added dtype cast to avoid errors
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
        A_p = A[N:N + self.nb_proxy].clone()
        A = A[:N]
        # print(A[0], A[0, N+ fLvec[0]], self.d(torch.unsqueeze(A[0], 0), torch.unsqueeze(A_p[fLvec[0]], 0)))
        # print( A[0,N+fLvec[0]],A[0,N+fLvec[1]])
        loss_intrinsic = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)  # Modified to avoid errors
        loss_context = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)  # Modified to avoid errors

        for i in range(N):
            loss_neg1_intrinsic = torch.zeros(1, dtype=fvec.dtype).cuda()
            loss_neg1_context = torch.zeros(1, dtype=fvec.dtype).cuda()
            dist_pos = self.d(torch.unsqueeze(A[i], 0), torch.unsqueeze(A_p[fLvec[i]], 0))

            for j in range(self.nb_proxy):

                if fLvec[i] != j:

                    val1_context = self.d(torch.unsqueeze(A[i], 0), torch.unsqueeze(A_p[j], 0)) - dist_pos
                    val1_intrinsic = A[i, N + j] - A[i, N + fLvec[i]]
                    if val1_context > 0:
                        loss_neg1_context += torch.exp(val1_context)
                    if val1_intrinsic > 0:
                        loss_neg1_intrinsic += torch.exp(val1_intrinsic)

            loss_intrinsic += torch.log(1.0 + loss_neg1_intrinsic)  # + torch.log(torch.add(loss_neg2_intrinsec, 1.0))
            loss_context += torch.log(1.0 + loss_neg1_context)  # + torch.log(torch.add(loss_neg2_context, 1.0))

        loss_intrinsic /= N
        loss_context /= N

        # print(loss_intrinsic, loss_context , loss_reg)
        # print(loss_intrinsic.item(), self.lambdaC*loss_context.item(), (loss_intrinsec + self.lambdaC*loss_context).item())
        return loss_intrinsic + self.lambdaC * loss_context


def loss_uncorrect_descriptors_dim():
    embeddings = torch.randn(10, 10)
    loss_fn = ManifoldLoss(l=999)
    loss_fn(embeddings)


class TestManifoldLoss(unittest.TestCase):
    def test_intrinsic_and_context_losses(self):
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            n_proxies = 3
            proxies = nn.Parameter(torch.randn(n_proxies, 2, device='cuda:0', dtype=dtype))
            alpha = 0.99
            original_loss_func = OriginalImplementationManifoldLoss(proxies, alpha=alpha, lambdaC=0)
            original_loss = original_loss_func(embeddings, labels)

            self.assertRaises(ValueError, lambda: ManifoldLoss(l=2, lambdaC=-999))
            self.assertRaises(AssertionError, loss_uncorrect_descriptors_dim)

            loss_func = ManifoldLoss(l=2, K=n_proxies, alpha=alpha, lambdaC=0, margin=0.)
            loss_func.proxies.data = proxies.data  # In order to have same initializations
            loss = loss_func(embeddings, indices_tuple=labels)

            rtol = 0.5 if dtype == torch.float16 else 4e-1

            self.assertTrue(torch.isclose(original_loss, loss, rtol=rtol))

    def test_with_original_implementation(self):
        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(5).tolist()
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor(torch.randint(0, 5, (5,)).tolist())

            n_proxies = 5
            proxies = nn.Parameter(torch.randn(n_proxies, 2, device='cuda:0', dtype=dtype))
            alpha = 0.99
            original_loss_func = OriginalImplementationManifoldLoss(proxies, alpha=alpha)
            original_loss = original_loss_func(embeddings, labels)

            loss_func = ManifoldLoss(l=2, K=n_proxies, alpha=alpha, margin=0.)  # Original implementation does
                                                                                # not consider margin                   # NoQA

            loss_func.proxies.data = proxies.data  # In order to have same initializations
            loss = loss_func(embeddings, indices_tuple=labels)

            rtol = 0.5 if dtype == torch.float16 else 4e-1

            self.assertTrue(torch.isclose(original_loss, loss, rtol=rtol))


if __name__ == '__main__':
    unittest.main()
