import unittest

import torch

from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity


class TestBatchedDistance(unittest.TestCase):
    def test_batched_distance(self):
        torch.manual_seed(1432)
        embedding_size = 128
        for dataset_size in [15, 149, 1501, 3199, 3200]:
            embeddings = torch.randn(
                dataset_size, embedding_size, dtype=torch.float64, device="cpu"
            )
            ref = torch.randn(
                dataset_size, embedding_size, dtype=torch.float64, device="cpu"
            )
            for batch_size in [32, 33, 99, 121, 128]:
                for dist_fn in [CosineSimilarity()]:
                    for use_ref in [False, True]:
                        distance = BatchedDistance(dist_fn, batch_size=batch_size)
                        if use_ref:
                            blocks = list(distance(embeddings, ref))
                            correct_mat = dist_fn(embeddings, ref)
                        else:
                            blocks = list(distance(embeddings))
                            correct_mat = dist_fn(embeddings)
                        starts = [start for _, start, _ in blocks]
                        self.assertEqual(
                            starts, list(range(0, dataset_size, batch_size))
                        )
                        mat = torch.cat([block for block, _, _ in blocks], dim=0)
                        self.assertTrue(torch.allclose(mat, correct_mat))
