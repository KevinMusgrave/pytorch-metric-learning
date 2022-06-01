import unittest

import torch

from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
    SNRDistance,
)
from pytorch_metric_learning.miners import BatchEasyHardMiner

from .. import TEST_DEVICE, TEST_DTYPES


class TestBatchEasyHardMinerLabels(unittest.TestCase):
    def test_labels(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(64, 256, device=TEST_DEVICE, dtype=dtype)
            labels = torch.randint(0, 10, size=(64,), device=TEST_DEVICE)
            for distance in [
                LpDistance,
                CosineSimilarity,
                DotProductSimilarity,
                SNRDistance,
            ]:
                for (pos_strategy, neg_strategy) in [
                    ("easy", "easy"),
                    ("easy", "semihard"),
                    ("easy", "hard"),
                    ("easy", "all"),
                    ("semihard", "easy"),
                    ("semihard", "hard"),
                    ("hard", "easy"),
                    ("hard", "semihard"),
                    ("hard", "hard"),
                    ("hard", "all"),
                    ("all", "easy"),
                    ("all", "hard"),
                    ("all", "all"),
                ]:
                    for pos_range, neg_range in [
                        (None, None),
                        ([0.2, 0.5], None),
                        (None, [0.2, 0.5]),
                    ]:
                        miner = BatchEasyHardMiner(
                            distance=distance(),
                            pos_strategy=pos_strategy,
                            neg_strategy=neg_strategy,
                            allowed_pos_range=pos_range,
                            allowed_neg_range=neg_range,
                        )
                        a1, p, a2, n = miner(embeddings, labels)
                        self.assertTrue(torch.all(labels[a1] == labels[p]))
                        self.assertTrue(not torch.any(labels[a2] == labels[n]))
