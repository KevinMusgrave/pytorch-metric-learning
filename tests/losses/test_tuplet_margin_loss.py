import math
import unittest

import numpy as np
import torch

from pytorch_metric_learning.losses import TupletMarginLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TestTupletMarginLoss(unittest.TestCase):
    def test_tuplet_margin_loss(self):
        margin, scale = 5, 64
        loss_func = TupletMarginLoss(margin=margin, scale=scale)

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [helpers.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()

            pos_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            neg_pairs = [
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (3, 1),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
            ]

            correct_total = torch.zeros((), dtype=torch.float64, device=TEST_DEVICE)
            embeddings = torch.nn.functional.normalize(embeddings.detach().double())
            for a1, p in pos_pairs:
                curr_loss = torch.zeros((), dtype=torch.float64, device=TEST_DEVICE)
                anchor1, positive = embeddings[a1], embeddings[p]
                ap_angle = torch.acos(torch.matmul(anchor1, positive))
                ap_cos = torch.cos(ap_angle - np.radians(margin))
                for a2, n in neg_pairs:
                    if a2 == a1:
                        anchor2, negative = embeddings[a2], embeddings[n]
                        an_cos = torch.matmul(anchor2, negative)
                        curr_loss += torch.exp(scale * (an_cos - ap_cos))

                curr_total = torch.log(1 + curr_loss)
                correct_total += curr_total

            correct_total /= len(pos_pairs)
            helpers.assert_close(loss, correct_total, dtype, atol=1e-8)

    def test_with_no_valid_pairs(self):
        loss_func = TupletMarginLoss(0.1, 64)
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [helpers.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0], dtype=torch.long).to(TEST_DEVICE)
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)


class TestTupletMarginDuplicates(unittest.TestCase):
    def test_endpoint_values_and_gradients_follow_the_angular_objective(self):
        labels = torch.tensor([0], device=TEST_DEVICE)
        reference_labels = torch.tensor([0, 1], device=TEST_DEVICE)
        query = torch.tensor([[1, 0]])
        for sign in (1, -1):
            reference = torch.tensor([[sign, 0], [math.cos(0.1), math.sin(0.1)]])
            for margin in (0, 5.73, 30):
                loss = TupletMarginLoss(margin=margin)
                positive = sign * math.cos(math.radians(margin))
                for dtype in TEST_DTYPES:
                    with self.subTest(sign=sign, margin=margin, dtype=dtype):
                        helpers.assert_matches_float64(
                            lambda q, r, loss=loss: loss(
                                q, labels, ref_emb=r, ref_labels=reference_labels
                            ),
                            lambda q, r, positive=positive: (
                                torch.nn.functional.softplus(
                                    64
                                    * (
                                        torch.sum(
                                            torch.nn.functional.normalize(q)
                                            * torch.nn.functional.normalize(r)[1]
                                        )
                                        - positive
                                    )
                                )
                            ),
                            [query, reference],
                            dtype,
                        )

    def test_zero_margin_preserves_near_endpoint_positive_gradients(self):
        labels = torch.tensor([0], device=TEST_DEVICE)
        reference_labels = torch.tensor([0, 1], device=TEST_DEVICE)
        query = torch.tensor([[1, 0]])
        reference = torch.tensor(
            [[math.cos(0.04), math.sin(0.04)], [math.cos(0.1), math.sin(0.1)]]
        )
        loss = TupletMarginLoss(margin=0, scale=16)
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                helpers.assert_matches_float64(
                    lambda q, r: loss(
                        q, labels, ref_emb=r, ref_labels=reference_labels
                    ),
                    lambda q, r: torch.nn.functional.cross_entropy(
                        16
                        * torch.nn.functional.normalize(q)
                        @ torch.nn.functional.normalize(r).T,
                        labels,
                    ),
                    [query, reference],
                    dtype,
                )

    def test_identical_positive_pair_has_finite_loss_and_gradient(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                embeddings = torch.randn(4, 8, device=TEST_DEVICE, dtype=dtype)
                embeddings = torch.cat([embeddings, embeddings[:1]]).requires_grad_()
                labels = torch.tensor([0, 1, 2, 3, 0], device=TEST_DEVICE)
                loss = TupletMarginLoss()(embeddings, labels)
                (grad,) = torch.autograd.grad(loss, embeddings)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(grad).all())


if __name__ == "__main__":
    unittest.main()
