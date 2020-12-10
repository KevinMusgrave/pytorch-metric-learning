import unittest
import torch
from .. import TEST_DTYPES, TEST_DEVICE
from pytorch_metric_learning.miners import BatchEasyHardMiner
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.distances import LpDistance
import numpy as np

class TestBatchEasyHardMiner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
        self.distance = LpDistance(normalize_embeddings=False)
        self.gt = {
            "batch_hard_hard" : {
                            "miner" : BatchEasyHardMiner(distance=self.distance,
                                                         positive_strategy=BatchEasyHardMiner.HARD, 
                                                         negative_strategy=BatchEasyHardMiner.HARD),
                            "pos_pair_dist" : 6,
                            "neg_pair_dist" : 1,
                            "expected" : {
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(TEST_DEVICE),
                                "correct_p" : [ torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]).to(TEST_DEVICE) ],
                                "correct_n" : [ torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]).to(TEST_DEVICE),
                                                torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]).to(TEST_DEVICE) ]
                            }
                        },
            "batch_easy_hard" : {
                            "miner" : BatchEasyHardMiner(distance=self.distance,
                                                         positive_strategy=BatchEasyHardMiner.EASY, 
                                                         negative_strategy=BatchEasyHardMiner.HARD),
                            "pos_pair_dist" : 1,
                            "neg_pair_dist" : 1,
                            "expected" : {
                                "correct_a" : torch.LongTensor(  [0, 1, 2, 3, 4, 6, 7, 8]).to(TEST_DEVICE),
                                "correct_p" : [ torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                                                torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE)],
                                "correct_n" : [ torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]).to(TEST_DEVICE),
                                                torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]).to(TEST_DEVICE) ]
                            }
                        },
            "batch_hard_easy" : {
                            "miner" : BatchEasyHardMiner(distance=self.distance,
                                                         positive_strategy=BatchEasyHardMiner.HARD, 
                                                         negative_strategy=BatchEasyHardMiner.EASY),
                            "pos_pair_dist" : 6,
                            "neg_pair_dist" : 8,
                            "expected" : {   
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(TEST_DEVICE),
                                "correct_p" : [ torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]).to(TEST_DEVICE) ],
                                "correct_n" : [ torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0]).to(TEST_DEVICE) ]
                            }
                        },
            "batch_easy_easy" : {
                            "miner" : BatchEasyHardMiner(distance=self.distance,
                                                         positive_strategy=BatchEasyHardMiner.EASY, 
                                                         negative_strategy=BatchEasyHardMiner.EASY),
                            "pos_pair_dist" : 1,
                            "neg_pair_dist" : 8,
                            "expected" : {   
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(TEST_DEVICE),
                                "correct_p" : [ torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                                                torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE)],
                                "correct_n" : [ torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0]).to(TEST_DEVICE) ]
                            }
                        },
            "batch_easy_easy_with_min_val" : {
                            "miner" : BatchEasyHardMiner(distance=self.distance,
                                                         positive_strategy=BatchEasyHardMiner.EASY, 
                                                         negative_strategy=BatchEasyHardMiner.EASY,
                                                         allowed_negative_range=[1,7],
                                                         allowed_positive_range=[1,7]
                                                         ),
                            "pos_pair_dist" : 1,
                            "neg_pair_dist" : 7,
                            "expected" : {   
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(TEST_DEVICE),
                                "correct_p" : [ torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]).to(TEST_DEVICE),
                                                torch.LongTensor([1, 0, 3, 2, 1, 7, 6, 7]).to(TEST_DEVICE)],
                                "correct_n" : [ torch.LongTensor([7, 8, 5, 0, 8, 0, 0, 1]).to(TEST_DEVICE) ]
                            }
                        }
        }

    def test_dist_mining(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.arange(9).type(dtype).unsqueeze(1).to(TEST_DEVICE)
            for miner in self.gt.keys():
                cfg = self.gt[miner]
                miner = cfg["miner"]
                a, p, n = miner.mine(embeddings, self.labels, embeddings, self.labels)
                self.helper(a, p, n, cfg["expected"])
                self.assertTrue(miner.pos_pair_dist == cfg["pos_pair_dist"])
                self.assertTrue(miner.neg_pair_dist == cfg["neg_pair_dist"])
    
    def helper(self, a, p, n, gt):
        self.assertTrue(torch.equal(a, gt["correct_a"]))
        self.assertTrue(any(torch.equal(p, cn) for cn in gt["correct_p"]))
        self.assertTrue(any(torch.equal(n, cn) for cn in gt["correct_n"]))

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()