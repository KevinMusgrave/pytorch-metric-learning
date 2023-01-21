import unittest

import torch

from pytorch_metric_learning.distances import BaseDistance, LpDistance


class CustomDistance(BaseDistance):
    def compute_mat(self, query_emb, ref_emb):
        pass

    def check_embeddings_ndim(self, query_emb, ref_emb):
        pass


class TestCustomEmbeddingNdim(unittest.TestCase):
    def test_custom_embedding_ndim(self):
        embeddings = torch.randn(32, 3, 128)

        dist_fn = LpDistance()

        with self.assertRaises(ValueError):
            dist_fn(embeddings)
