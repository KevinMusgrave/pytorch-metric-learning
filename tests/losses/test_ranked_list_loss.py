import unittest

import numpy as np
import torch

from pytorch_metric_learning.losses import RankedListLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestRankedListLoss(unittest.TestCase):
    def test_ranked_list_loss_simpler(self):
        batch_size = 32
        embedding_size = 64
        for dtype in TEST_DTYPES:
            # test multiple times
            for _ in range(2):
                embeddings = torch.randn(
                    batch_size,
                    embedding_size,
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.randint(0, 5, size=(batch_size,))
                normalized_embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                )
                n = len(embeddings)
                for Tp, lam, margin in zip(
                    [0, 0.5, 3, np.inf], [0, 0.5, 0.7, 0.9], [0, 0.4, 0.8, 1.2]
                ):
                    alpha = 1 - margin / 2
                    Tn = Tp
                    loss_func = RankedListLoss(
                        margin=margin, Tn=Tn, imbalance=lam, alpha=alpha, Tp=Tp
                    )
                    L_RLL = torch.zeros(
                        n,
                    ).to(dtype=dtype)
                    for i in range(n):
                        w_p = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        w_n = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        L_P = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        L_N = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        for j in range(n):
                            if i == j:
                                continue

                            d_ij = (
                                torch.sum(
                                    (
                                        normalized_embeddings[i, :]
                                        - normalized_embeddings[j, :]
                                    )
                                    ** 2
                                )
                                ** 0.5
                            )
                            if labels[j] == labels[i] and d_ij > alpha - margin:
                                w_p[j] = torch.exp(Tp * (d_ij - (alpha - margin)))
                                L_P[j] = d_ij - (alpha - margin)
                            elif labels[j] != labels[i] and d_ij < alpha:
                                w_n[j] = torch.exp(Tn * (alpha - d_ij))
                                L_N[j] = alpha - d_ij
                        L_P = (
                            torch.sum(w_p * L_P) / torch.sum(w_p)
                            if torch.sum(w_p) > 0
                            else 0
                        )
                        L_N = (
                            torch.sum(w_n * L_N) / torch.sum(w_n)
                            if torch.sum(w_n) > 0
                            else 0
                        )
                        L_RLL[i] = (1 - lam) * L_P + lam * L_N
                    correct_loss = torch.mean(L_RLL)
                    loss = loss_func(embeddings, labels)

                    rtol = 1e-2 if dtype == torch.float16 else 1e-5
                    self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

                    loss.backward()

    def test_ranked_list_loss(self):
        batch_size = 32
        embedding_size = 64
        for dtype in TEST_DTYPES:
            # test multiple times
            for _ in range(2):
                embeddings = torch.randn(
                    batch_size,
                    embedding_size,
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.randint(0, 5, size=(batch_size,))
                n = len(embeddings)
                for Tn, Tp, alpha, lam, margin in zip(
                    [0.3, 0.8, 2, 10],
                    [0, 0.5, 3, np.inf],
                    [0.3, 0.8, 1, 3],
                    [0, 0.5, 0.7, 0.9],
                    [0, 0.4, 0.8, 1.2],
                ):
                    loss_func = RankedListLoss(
                        margin=margin, Tn=Tn, imbalance=lam, alpha=alpha, Tp=Tp
                    )
                    L_RLL = torch.zeros(
                        n,
                    ).to(dtype=dtype)
                    for i in range(n):
                        w_p = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        w_n = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        L_P = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        L_N = torch.zeros(
                            n,
                        ).to(dtype=dtype)
                        for j in range(n):
                            d_ij = (
                                torch.sum((embeddings[i, :] - embeddings[j, :]) ** 2)
                                ** 0.5
                            )
                            if (
                                i != j
                                and labels[j] == labels[i]
                                and d_ij > alpha - margin
                            ):
                                w_p[j] = torch.exp(Tp * (d_ij - (alpha - margin)))
                                L_P[j] = d_ij - (alpha - margin)
                            elif labels[j] == labels[i] and d_ij < alpha:
                                w_n[j] = torch.exp(Tn * (alpha - d_ij))
                                L_N[j] = alpha - d_ij
                        L_P = torch.sum(w_p * L_P) / torch.sum(w_p)
                        L_N = torch.sum(w_n * L_N) / torch.sum(w_n)
                        L_RLL[i] = (1 - lam) * L_P + lam * L_N
                    correct_loss = torch.mean(L_RLL)
                    loss = loss_func(embeddings, labels)

                    rtol = 1e-2 if dtype == torch.float16 else 1e-5
                    self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

                    loss.backward()

    def test_assertion_raises(self):
        with self.assertRaises(AssertionError):
            _ = RankedListLoss(margin=1, Tn=0, imbalance=-1)
            _ = RankedListLoss(margin=1, Tn=0, imbalance=2)
