import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.miners import BatchEasyHardMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES, WITH_COLLECT_STATS


class TestBatchEasyHardMiner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
        self.a1_idx, self.p_idx, self.a2_idx, self.n_idx = lmu.get_all_pairs_indices(
            self.labels
        )
        self.distance = LpDistance(normalize_embeddings=False)
        self.gt = {
            "batch_semihard_hard": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.SEMIHARD,
                    neg_strategy=BatchEasyHardMiner.HARD,
                ),
                "easiest_triplet": -1,
                "hardest_triplet": -1,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 2,
                "easiest_neg_pair": 3,
                "hardest_neg_pair": 2,
                "expected": {
                    "correct_a": torch.LongTensor([0, 7, 8]).to(TEST_DEVICE),
                    "correct_p": [
                        torch.LongTensor([1, 6, 6]).to(TEST_DEVICE),
                        torch.LongTensor([1, 8, 6]).to(TEST_DEVICE),
                    ],
                    "correct_n": [
                        torch.LongTensor([2, 5, 5]).to(TEST_DEVICE),
                        torch.LongTensor([2, 5, 5]).to(TEST_DEVICE),
                    ],
                },
            },
            "batch_hard_semihard": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.HARD,
                    neg_strategy=BatchEasyHardMiner.SEMIHARD,
                ),
                "easiest_triplet": -1,
                "hardest_triplet": -1,
                "easiest_pos_pair": 3,
                "hardest_pos_pair": 6,
                "easiest_neg_pair": 7,
                "hardest_neg_pair": 4,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 6, 7, 8]).to(TEST_DEVICE),
                    "correct_p": [torch.LongTensor([4, 4, 2, 2, 2]).to(TEST_DEVICE)],
                    "correct_n": [
                        torch.LongTensor([5, 5, 1, 1, 1]).to(TEST_DEVICE),
                    ],
                },
            },
            "batch_easy_semihard": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.EASY,
                    neg_strategy=BatchEasyHardMiner.SEMIHARD,
                ),
                "easiest_triplet": -2,
                "hardest_triplet": -1,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 3,
                "easiest_neg_pair": 4,
                "hardest_neg_pair": 2,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE),
                    ],
                    "correct_n": [
                        torch.LongTensor([2, 3, 0, 1, 8, 4, 5, 5]).to(TEST_DEVICE),
                        torch.LongTensor([2, 3, 4, 1, 8, 4, 5, 5]).to(TEST_DEVICE),
                        torch.LongTensor([2, 3, 0, 5, 8, 4, 5, 5]).to(TEST_DEVICE),
                        torch.LongTensor([2, 3, 4, 5, 8, 4, 5, 5]).to(TEST_DEVICE),
                    ],
                },
            },
            "batch_hard_hard": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.HARD,
                    neg_strategy=BatchEasyHardMiner.HARD,
                ),
                "easiest_triplet": 2,
                "hardest_triplet": 5,
                "easiest_pos_pair": 3,
                "hardest_pos_pair": 6,
                "easiest_neg_pair": 3,
                "hardest_neg_pair": 1,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]).to(TEST_DEVICE)
                    ],
                    "correct_n": [
                        torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]).to(TEST_DEVICE),
                        torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]).to(TEST_DEVICE),
                    ],
                },
            },
            "batch_easy_hard": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.EASY,
                    neg_strategy=BatchEasyHardMiner.HARD,
                ),
                "easiest_triplet": -2,
                "hardest_triplet": 2,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 3,
                "easiest_neg_pair": 3,
                "hardest_neg_pair": 1,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE),
                    ],
                    "correct_n": [
                        torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]).to(TEST_DEVICE),
                        torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]).to(TEST_DEVICE),
                    ],
                },
            },
            "batch_hard_easy": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.HARD,
                    neg_strategy=BatchEasyHardMiner.EASY,
                ),
                "easiest_triplet": -4,
                "hardest_triplet": 3,
                "easiest_pos_pair": 3,
                "hardest_pos_pair": 6,
                "easiest_neg_pair": 8,
                "hardest_neg_pair": 3,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]).to(TEST_DEVICE)
                    ],
                    "correct_n": [
                        torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0]).to(TEST_DEVICE)
                    ],
                },
            },
            "batch_easy_easy": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.EASY,
                    neg_strategy=BatchEasyHardMiner.EASY,
                ),
                "easiest_triplet": -7,
                "hardest_triplet": -1,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 3,
                "easiest_neg_pair": 8,
                "hardest_neg_pair": 3,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE),
                    ],
                    "correct_n": [
                        torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0]).to(TEST_DEVICE)
                    ],
                },
            },
            "batch_easy_easy_with_min_val": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.EASY,
                    neg_strategy=BatchEasyHardMiner.EASY,
                    allowed_neg_range=[1, 7],
                    allowed_pos_range=[1, 7],
                ),
                "easiest_triplet": -6,
                "hardest_triplet": -1,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 3,
                "easiest_neg_pair": 7,
                "hardest_neg_pair": 3,
                "expected": {
                    "correct_a": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE),
                    ],
                    "correct_n": [
                        torch.LongTensor([7, 8, 5, 0, 8, 0, 0, 1]).to(TEST_DEVICE)
                    ],
                },
            },
            "batch_easy_all": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.EASY,
                    neg_strategy=BatchEasyHardMiner.ALL,
                ),
                "easiest_triplet": 0,
                "hardest_triplet": 0,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 3,
                "easiest_neg_pair": 8,
                "hardest_neg_pair": 1,
                "expected": {
                    "correct_a1": torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_p": [
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                        torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE),
                    ],
                    "correct_a2": self.a2_idx,
                    "correct_n": [self.n_idx],
                },
            },
            "batch_all_easy": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.ALL,
                    neg_strategy=BatchEasyHardMiner.EASY,
                ),
                "easiest_triplet": 0,
                "hardest_triplet": 0,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 6,
                "easiest_neg_pair": 8,
                "hardest_neg_pair": 3,
                "expected": {
                    "correct_a1": self.a1_idx,
                    "correct_p": [self.p_idx],
                    "correct_a2": torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).to(
                        TEST_DEVICE
                    ),
                    "correct_n": [
                        torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0, 0]).to(TEST_DEVICE),
                    ],
                },
            },
            "batch_all_all": {
                "miner": BatchEasyHardMiner(
                    distance=self.distance,
                    pos_strategy=BatchEasyHardMiner.ALL,
                    neg_strategy=BatchEasyHardMiner.ALL,
                ),
                "easiest_triplet": 0,
                "hardest_triplet": 0,
                "easiest_pos_pair": 1,
                "hardest_pos_pair": 6,
                "easiest_neg_pair": 8,
                "hardest_neg_pair": 1,
                "expected": {
                    "correct_a1": self.a1_idx,
                    "correct_p": [self.p_idx],
                    "correct_a2": self.a2_idx,
                    "correct_n": [self.n_idx],
                },
            },
        }

    def test_dist_mining(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.arange(9).type(dtype).unsqueeze(1).to(TEST_DEVICE)
            for miner in self.gt.keys():
                cfg = self.gt[miner]
                miner = cfg["miner"]
                a1, p, a2, n = miner.mine(
                    embeddings, self.labels, embeddings, self.labels
                )
                self.helper(a1, p, a2, n, cfg["expected"])
                if WITH_COLLECT_STATS:
                    self.assertTrue(miner.easiest_triplet == cfg["easiest_triplet"])
                    self.assertTrue(miner.hardest_triplet == cfg["hardest_triplet"])
                    self.assertTrue(miner.easiest_pos_pair == cfg["easiest_pos_pair"])
                    self.assertTrue(miner.hardest_pos_pair == cfg["hardest_pos_pair"])
                    self.assertTrue(miner.easiest_neg_pair == cfg["easiest_neg_pair"])
                    self.assertTrue(miner.hardest_neg_pair == cfg["hardest_neg_pair"])

    def test_strategy_assertion(self):
        self.assertRaises(ValueError, lambda: BatchEasyHardMiner(pos_strategy="blah"))
        self.assertRaises(
            ValueError,
            lambda: BatchEasyHardMiner(
                pos_strategy="semihard", neg_strategy="semihard"
            ),
        )
        self.assertRaises(
            ValueError,
            lambda: BatchEasyHardMiner(pos_strategy="all", neg_strategy="semihard"),
        )
        self.assertRaises(
            ValueError,
            lambda: BatchEasyHardMiner(pos_strategy="semihard", neg_strategy="all"),
        )

    def helper(self, a1, p, a2, n, gt):
        try:
            self.assertTrue(torch.equal(a1, gt["correct_a"]))
            self.assertTrue(torch.equal(a2, gt["correct_a"]))
            self.assertTrue(any(torch.equal(p, cn) for cn in gt["correct_p"]))
            self.assertTrue(any(torch.equal(n, cn) for cn in gt["correct_n"]))
        except Exception:
            self.assertTrue(torch.equal(a1, gt["correct_a1"]))
            self.assertTrue(torch.equal(a2, gt["correct_a2"]))
            self.assertTrue(any(torch.equal(p, cn) for cn in gt["correct_p"]))
            self.assertTrue(any(torch.equal(n, cn) for cn in gt["correct_n"]))

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
