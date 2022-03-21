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
            # test on cpu because gpu is unreliable
            embeddings = torch.randn(dataset_size, embedding_size, dtype=torch.float64)
            ref = torch.randn(dataset_size, embedding_size, dtype=torch.float64)
            for batch_size in [32, 33, 99, 121, 128]:
                for dist_fn in [CosineSimilarity()]:
                    for use_ref in [False, True]:
                        mat = []
                        distance = BatchedDistance(dist_fn, collect_fn(mat), batch_size)
                        if use_ref:
                            distance(embeddings, ref)
                            correct_mat = dist_fn(embeddings, ref)
                        else:
                            distance(embeddings)
                            correct_mat = dist_fn(embeddings)
                            normalized_emb = torch.nn.functional.normalize(embeddings)
                        mat = torch.cat(mat, dim=0)
                        self.assertTrue(torch.allclose(mat, correct_mat))

    def test_attr(self):
        dist_fn = CosineSimilarity()
        distance = BatchedDistance(dist_fn, None)
        # via getattr
        for attr in ["is_inverted", "normalize_embeddings", "p", "power"]:
            self.assertTrue(getattr(distance, attr) == getattr(dist_fn, attr))

        # using dot
        self.assertTrue(dist_fn.is_inverted == distance.is_inverted)
        self.assertTrue(dist_fn.normalize_embeddings == distance.normalize_embeddings)
        self.assertTrue(dist_fn.p == distance.p)
        self.assertTrue(dist_fn.power == distance.power)
