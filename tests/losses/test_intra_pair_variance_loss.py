import unittest

import torch

from pytorch_metric_learning.losses import IntraPairVarianceLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import angle_to_coord


class TestIntraPairVarianceLoss(unittest.TestCase):
    def test_intra_pair_variance_loss(self):
        pos_eps, neg_eps = 0.01, 0.02
        loss_func = IntraPairVarianceLoss(pos_eps, neg_eps)

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

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

            zero = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
            pos_total, neg_total = zero.clone(), zero.clone()
            mean_pos, mean_neg = zero.clone(), zero.clone()
            embeddings = torch.nn.functional.normalize(embeddings)
            for a, p in pos_pairs:
                mean_pos += torch.matmul(embeddings[a], embeddings[p])
            for a, n in neg_pairs:
                mean_neg += torch.matmul(embeddings[a], embeddings[n])
            mean_pos /= len(pos_pairs)
            mean_neg /= len(neg_pairs)

            for a, p in pos_pairs:
                pos_total += (
                    torch.relu(
                        (1 - pos_eps) * mean_pos
                        - torch.matmul(embeddings[a], embeddings[p])
                    )
                    ** 2
                )
            for a, n in neg_pairs:
                neg_total += (
                    torch.relu(
                        torch.matmul(embeddings[a], embeddings[n])
                        - (1 + neg_eps) * mean_neg
                    )
                    ** 2
                )

            pos_total /= len(pos_pairs)
            neg_total /= len(neg_pairs)
            correct_total = pos_total + neg_total
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))

    def test_with_no_valid_pairs(self):
        loss_func = IntraPairVarianceLoss(0.01, 0.01)
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0], dtype=torch.long)
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)
