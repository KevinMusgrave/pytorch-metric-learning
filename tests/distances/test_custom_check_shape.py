import unittest

import torch

from pytorch_metric_learning.distances import BaseDistance, LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss


class CustomDistance(BaseDistance):
    def compute_mat(self, query_emb, ref_emb):
        return torch.randn(query_emb.shape[0], ref_emb.shape[0])

    def check_shapes(self, query_emb, ref_emb):
        pass


class TestCustomCheckShape(unittest.TestCase):
    def test_custom_embedding_ndim(self):
        embeddings = torch.randn(32, 3, 128)
        labels = torch.randint(0, 10, size=(32,))
        loss_fn = TripletMarginLoss(distance=LpDistance())

        with self.assertRaises(ValueError):
            loss_fn(embeddings, labels)

        loss_fn = TripletMarginLoss(distance=CustomDistance())
        loss_fn(embeddings, labels)
