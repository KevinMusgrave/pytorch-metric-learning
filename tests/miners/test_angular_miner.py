import unittest

import numpy as np
import torch

from pytorch_metric_learning.miners import AngularMiner

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def planar_embeddings(num_points, dtype):
    return torch.tensor(
        [helpers.angle_to_coord(a) for a in torch.arange(0, num_points)],
        requires_grad=True,
        dtype=dtype,
    ).to(TEST_DEVICE)


def anchor_positive_negative_triplets(labels):
    num_points = len(labels)
    for i in range(num_points):
        for j in range(num_points):
            if j == i or labels[j] != labels[i]:
                continue
            for k in range(num_points):
                if k not in (i, j) and labels[k] != labels[j]:
                    yield i, j, k


def triplet_angles(embeddings, labels):
    embeddings = torch.nn.functional.normalize(embeddings.detach().double(), dim=1)
    angles = []
    for i, j, k in anchor_positive_negative_triplets(labels):
        anchor, positive, negative = embeddings[i], embeddings[j], embeddings[k]
        center = (anchor + positive) / 2
        ap_dist = torch.linalg.vector_norm(anchor - positive)
        nc_dist = torch.linalg.vector_norm(center - negative)
        angles.append((i, j, k, torch.atan(ap_dist / (2 * nc_dist))))
    return angles


class TestAngularMiner(unittest.TestCase):
    def test_angular_miner(self):
        for dtype in TEST_DTYPES:
            embeddings = planar_embeddings(16, dtype)
            labels = torch.randint(low=0, high=2, size=(16,))
            angles = triplet_angles(embeddings, labels)

            for angle_in_degrees in range(0, 70, 10):
                miner = AngularMiner(angle_in_degrees)
                angle_in_radians = np.radians(angle_in_degrees)
                correct_triplets = {
                    (i, j, k) for i, j, k, angle in angles if angle > angle_in_radians
                }
                a1, p1, n1 = miner(embeddings, labels)
                mined_triplets = {
                    (a.item(), p.item(), n.item())
                    for a, p, n in zip(a1, p1, n1, strict=True)
                }
                self.assertTrue(mined_triplets == correct_triplets)

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
