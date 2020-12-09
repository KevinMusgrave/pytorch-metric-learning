import unittest
import torch
from pytorch_metric_learning.miners import BatchEasyHardMiner
from pytorch_metric_learning.utils import common_functions as c_f
import numpy as np

class TestBatchEasyHardMiner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
        self.embeddings = torch.arange(9).float().unsqueeze(1)

        self.gt = {
            "batch_hard_hard" : {
                            "miner" : BatchEasyHardMiner(positive_strategy=BatchEasyHardMiner.HARD, 
                                                         negative_strategy=BatchEasyHardMiner.HARD,
                                                         use_similarity=False, 
                                                         normalize_embeddings=False),
                            "pos_pair_dist" : 6,
                            "neg_pair_dist" : 1,
                            "expected" : {
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]),
                                "correct_p" : torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]),
                                "correct_n" : [ torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]),
                                                torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]) ]
                            }
                        },
            "batch_easy_hard" : {
                            "miner" : BatchEasyHardMiner(positive_strategy=BatchEasyHardMiner.EASY, 
                                                         negative_strategy=BatchEasyHardMiner.HARD,
                                                         use_similarity=False, 
                                                         normalize_embeddings=False),
                            "pos_pair_dist" : 1,
                            "neg_pair_dist" : 1,
                            "expected" : {
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]),
                                "correct_p" : torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]),
                                "correct_n" : [ torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]),
                                                torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]) ]
                            }
                        },
            "batch_hard_easy" : {
                            "miner" : BatchEasyHardMiner(positive_strategy=BatchEasyHardMiner.HARD, 
                                                         negative_strategy=BatchEasyHardMiner.EASY,
                                                         use_similarity=False, 
                                                         normalize_embeddings=False),
                            "pos_pair_dist" : 6,
                            "neg_pair_dist" : 8,
                            "expected" : {   
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]),
                                "correct_p" : torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]),
                                "correct_n" : [ torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0]) ]
                            }
                        },
            "batch_easy_easy" : {
                            "miner" : BatchEasyHardMiner(positive_strategy=BatchEasyHardMiner.EASY, 
                                                         negative_strategy=BatchEasyHardMiner.EASY,
                                                         use_similarity=False, 
                                                         normalize_embeddings=False),
                            "pos_pair_dist" : 1,
                            "neg_pair_dist" : 8,
                            "expected" : {   
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]),
                                "correct_p" : torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]),
                                "correct_n" : [ torch.LongTensor([8, 8, 5, 0, 8, 0, 0, 0]) ]
                            }
                        },
            "batch_easy_easy_with_min_val" : {
                            "miner" : BatchEasyHardMiner(positive_strategy=BatchEasyHardMiner.EASY, 
                                                         negative_strategy=BatchEasyHardMiner.EASY,
                                                         use_similarity=False, 
                                                         normalize_embeddings=False,
                                                         allowed_negative_range=[1,7],
                                                         allowed_positive_range=[1,7]
                                                         ),
                            "pos_pair_dist" : 1,
                            "neg_pair_dist" : 7,
                            "expected" : {   
                                "correct_a" : torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]),
                                "correct_p" : torch.LongTensor([1, 0, 3, 2, 1, 7, 8, 7]),
                                "correct_n" : [ torch.LongTensor([7, 8, 5, 0, 8, 0, 0, 1]) ]
                            }
                        }
        }

    def test_dist_mining(self):
         
        for miner in self.gt.keys():
            cfg = self.gt[miner]
            miner = cfg["miner"]
            a, p, n = miner.mine(self.embeddings, self.labels, self.embeddings, self.labels)
            self.helper(a, p, n, cfg["expected"])
            self.assertTrue(miner.pos_pair_dist == cfg["pos_pair_dist"])
            self.assertTrue(miner.neg_pair_dist == cfg["neg_pair_dist"])
    
    def helper(self, a, p, n, gt):
        self.assertTrue(torch.equal(a, gt["correct_a"]))
        self.assertTrue(torch.equal(p, gt["correct_p"]))
        self.assertTrue(any(torch.equal(n, cn) for cn in gt["correct_n"]))

if __name__ == '__main__':
    unittest.main()