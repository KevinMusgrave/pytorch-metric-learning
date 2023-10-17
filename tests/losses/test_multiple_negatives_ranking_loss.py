import unittest

import torch

from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import MultipleNegativesRankingLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner

from .. import TEST_DEVICE, TEST_DTYPES


class TestTripletMarginLoss(unittest.TestCase):
    def test_multiple_negatives_ranking_loss(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.LongTensor([0, 0, 1, 1, 1])
            self.helper(embeddings, labels, None, dtype)

    def test_multiple_negatives_ranking_loss_with_miner(self):
        miner = BatchEasyHardMiner(pos_strategy="hard", neg_strategy="all")
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.LongTensor([0, 0, 1, 1, 1])
            triplets = miner(embeddings, labels)
            self.helper(embeddings, labels, triplets, dtype)

    def helper(
        self, embeddings, labels, triplets, dtype, ref_emb=None, ref_labels=None
    ):
        loss_funcA = MultipleNegativesRankingLoss()
        loss_funcB = MultipleNegativesRankingLoss(scale=0.5)
        loss_funcC = MultipleNegativesRankingLoss(
            distance=DotProductSimilarity(normalize_embeddings=False)
        )
        [lossA, lossB, lossC] = [
            x(
                embeddings,
                labels,
                ref_emb=ref_emb,
                ref_labels=ref_labels,
                indices_tuple=triplets,
            )
            for x in [loss_funcA, loss_funcB, loss_funcC]
        ]
        self.assertTrue(lossC > lossA > lossB)

    def test_calculate_anchor_positive_loss(self):
        mnrl = MultipleNegativesRankingLoss()
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            anchor_idx = torch.arange(5, dtype=torch.long)
            positive_idx = [1, 0, 3, 2, 4]
            negative_idx = [[2, 3, 4], [2, 3, 4], [0, 1], [0, 1], [0, 1]]
            for idx in range(5):
                loss = mnrl.calculate_anchor_positive_loss(
                    embeddings, anchor_idx[idx], positive_idx[idx], negative_idx[idx]
                )
                self.assertTrue(loss.item() >= 0)
                self.assertTrue(loss.requires_grad)

    def test_get_logits(self):
        mnrl = MultipleNegativesRankingLoss()
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            anchor_idx = torch.arange(5, dtype=torch.long)
            positive_idx = [1, 0, 3, 2, 4]
            negative_idx = [[2, 3, 4], [2, 3, 4], [0, 1], [0, 1], [0, 1]]
            for idx in range(5):
                logits = mnrl.get_logits(
                    embeddings, anchor_idx[idx], positive_idx[idx], negative_idx[idx]
                )
                self.assertEqual(logits.shape, (len(negative_idx[idx]) + 1,))
                self.assertTrue(logits.requires_grad)
