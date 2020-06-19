import unittest
import torch
from pytorch_metric_learning.miners import PairMarginMiner
from pytorch_metric_learning.utils import common_functions as c_f

class TestPairMarginMiner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dist_miner = PairMarginMiner(pos_margin=4, neg_margin=4, use_similarity=False, normalize_embeddings=False)
        self.normalized_dist_miner = PairMarginMiner(pos_margin=1.29, neg_margin=1.28, use_similarity=False, normalize_embeddings=True)
        self.normalized_dist_miner_squared = PairMarginMiner(pos_margin=1.66, neg_margin=1.64, use_similarity=False, normalize_embeddings=True, squared_distances=True)
        self.sim_miner = PairMarginMiner(pos_margin=0.17, neg_margin=0.18, use_similarity=True, normalize_embeddings=True)
        self.labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
        self.correct_a1 = torch.LongTensor([2, 2, 3, 7, 8, 8])
        self.correct_p = torch.LongTensor([7, 8, 8, 2, 2, 3])
        self.correct_a2 = torch.LongTensor([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8])
        self.correct_n = torch.LongTensor([2, 3, 2, 3, 0, 1, 4, 5, 0, 1, 4, 5, 2, 3, 5, 6, 7, 2, 3, 4, 6, 7, 8, 4, 5, 4, 5, 5])

    def test_dist_mining(self):
        embeddings = torch.arange(9).float().unsqueeze(1)
        a1, p, a2, n = self.dist_miner(embeddings, self.labels)
        self.helper(a1, p, a2, n, self.dist_miner)

    def test_normalized_dist_mining(self):
        angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in angles])
        a1, p, a2, n = self.normalized_dist_miner(embeddings, self.labels)
        self.helper(a1, p, a2, n, self.normalized_dist_miner)

    def test_normalized_dist_squared_mining(self):
        angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in angles])
        a1, p, a2, n = self.normalized_dist_miner_squared(embeddings, self.labels)
        self.helper(a1, p, a2, n, self.normalized_dist_miner_squared)    

    def test_sim_mining(self):
        angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in angles])
        a1, p, a2, n = self.sim_miner(embeddings, self.labels)
        self.helper(a1, p, a2, n, self.sim_miner)

    def helper(self, a1, p, a2, n, miner):
        self.assertTrue(torch.equal(a1, self.correct_a1))
        self.assertTrue(torch.equal(p, self.correct_p))
        self.assertTrue(torch.equal(a2, self.correct_a2))
        self.assertTrue(torch.equal(n, self.correct_n))
        self.assertTrue(miner.pos_pair_dist>0)
        self.assertTrue(miner.neg_pair_dist>0)

    def test_empty_output(self):
        batch_size = 32
        embeddings = torch.randn(batch_size, 64)
        labels = torch.arange(batch_size)
        for miner in [self.dist_miner, self.normalized_dist_miner, self.normalized_dist_miner_squared, self.sim_miner]:
            a1, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a1)==0)
            self.assertTrue(len(p)==0)
            self.assertTrue(miner.pos_pair_dist==0)