import unittest

import numpy as np
import torch

from pytorch_metric_learning.miners import AngularMiner

from .. import TEST_DEVICE, TEST_DTYPES, WITH_COLLECT_STATS
from ..zzz_testing_utils import testing_utils


class TestAngularMiner(unittest.TestCase):
    def test_angular_miner(self):
        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 16)
            embeddings = torch.tensor(
                [testing_utils.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=2, size=(16,))
            triplets = []
            for i in range(len(embeddings)):
                anchor, anchor_label = embeddings[i], labels[i]
                for j in range(len(embeddings)):
                    if j == i:
                        continue
                    positive, positive_label = embeddings[j], labels[j]
                    center = (anchor + positive) / 2
                    if positive_label == anchor_label:
                        ap_dist = torch.nn.functional.pairwise_distance(
                            anchor.unsqueeze(0), positive.unsqueeze(0), 2
                        )
                        for k in range(len(embeddings)):
                            if k == j or k == i:
                                continue
                            negative, negative_label = embeddings[k], labels[k]
                            if negative_label != positive_label:
                                nc_dist = torch.nn.functional.pairwise_distance(
                                    center.unsqueeze(0), negative.unsqueeze(0), 2
                                )
                                angle = torch.atan(ap_dist / (2 * nc_dist))
                                triplets.append((i, j, k, angle))

            for angle_in_degrees in range(0, 70, 10):
                miner = AngularMiner(angle_in_degrees)
                angle_in_radians = np.radians(angle_in_degrees)
                correct = []
                for i, j, k, angle in triplets:
                    if angle > angle_in_radians:
                        correct.append((i, j, k))
                correct_triplets = set(correct)
                a1, p1, n1 = miner(embeddings, labels)
                mined_triplets = set(
                    [(a.item(), p.item(), n.item()) for a, p, n in zip(a1, p1, n1)]
                )
                self.assertTrue(mined_triplets == correct_triplets)
                testing_utils.is_not_none_if_condition(
                    self,
                    miner,
                    [
                        "average_angle",
                        "average_angle_above_threshold",
                        "average_angle_below_threshold",
                        "min_angle",
                        "max_angle",
                        "std_of_angle",
                    ],
                    WITH_COLLECT_STATS,
                )

    def test_empty_output(self):
        for dtype in TEST_DTYPES:
            miner = AngularMiner(35)
            batch_size = 32
            embeddings = torch.randn(batch_size, 64).to(TEST_DEVICE).type(dtype)
            labels = torch.arange(batch_size)
            a, p, n = miner(embeddings, labels)
            self.assertTrue(len(a) == 0)
            self.assertTrue(len(p) == 0)
            self.assertTrue(len(n) == 0)
