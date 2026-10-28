import unittest

import torch

from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity


def collect_fn(all_mat):
    def fn(mat, *_):
        all_mat.append(mat)

    return fn


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
                        mat = []
                        distance = BatchedDistance(dist_fn, batch_size=batch_size)
                        if use_ref:
                            distance(embeddings, ref, iter_fn=collect_fn(mat))
                            correct_mat = dist_fn(embeddings, ref)
                        else:
                            distance(embeddings, iter_fn=collect_fn(mat))
                            correct_mat = dist_fn(embeddings)
                        mat = torch.cat(mat, dim=0)
                        self.assertTrue(torch.allclose(mat, correct_mat))
