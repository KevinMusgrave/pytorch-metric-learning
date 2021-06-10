import unittest

import numpy as np
import torch

from pytorch_metric_learning.losses import AngularLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES, WITH_COLLECT_STATS
from ..zzz_testing_utils import testing_utils


class TestAngularLoss(unittest.TestCase):
    def test_angular_loss(self):
        loss_func = AngularLoss(alpha=40)

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            loss = loss_func(embeddings, labels)
            loss.backward()
            sq_tan_alpha = (
                torch.tan(torch.tensor(np.radians(40), dtype=dtype).to(TEST_DEVICE))
                ** 2
            )
            triplets = [
                (0, 1, 2),
                (0, 1, 3),
                (0, 1, 4),
                (1, 0, 2),
                (1, 0, 3),
                (1, 0, 4),
                (2, 3, 0),
                (2, 3, 1),
                (2, 3, 4),
                (3, 2, 0),
                (3, 2, 1),
                (3, 2, 4),
            ]

            correct_losses = [0, 0, 0, 0]
            for a, p, n in triplets:
                anchor, positive, negative = embeddings[a], embeddings[p], embeddings[n]
                exponent = 4 * sq_tan_alpha * torch.matmul(
                    anchor + positive, negative
                ) - 2 * (1 + sq_tan_alpha) * torch.matmul(anchor, positive)
                correct_losses[a] += torch.exp(exponent)
            total_loss = 0
            for c in correct_losses:
                total_loss += torch.log(1 + c)
            total_loss /= len(correct_losses)
            self.assertTrue(torch.isclose(loss, total_loss))

            testing_utils.is_not_none_if_condition(
                self, loss_func, ["average_angle"], WITH_COLLECT_STATS
            )

    def test_with_no_valid_triplets(self):
        loss_func = AngularLoss(alpha=40)
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 1, 2, 3, 4])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)
