import unittest
import torch
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.utils import common_functions as c_f

class TestBatchHardMiner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dist_miner = BatchHardMiner(use_similarity=False, normalize_embeddings=False)
        self.sim_miner = BatchHardMiner(use_similarity=True, normalize_embeddings=True)
        self.labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
        self.correct_a = torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8])
        self.correct_p = torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2])
        self.correct_n = [torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]), torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5])]

    def test_dist_mining(self):
        embeddings = torch.arange(9).float().unsqueeze(1)
        a, p, n = self.dist_miner(embeddings, self.labels)
        self.helper(a, p, n)

    def test_sim_mining(self):
        angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in angles])
        a, p, n = self.sim_miner(embeddings, self.labels)
        self.helper(a, p, n)

    def helper(self, a, p, n):
        self.assertTrue(torch.equal(a, self.correct_a))
        self.assertTrue(torch.equal(p, self.correct_p))
        self.assertTrue(any(torch.equal(n, cn) for cn in self.correct_n))

    def test_empty_output(self):
        batch_size = 32
        embeddings = torch.randn(batch_size, 64)
        labels = torch.arange(batch_size)
        a, p, n = self.dist_miner(embeddings, labels)
        self.assertTrue(len(a)==0)
        self.assertTrue(len(p)==0)
        self.assertTrue(len(n)==0)

        a, p, n = self.sim_miner(embeddings, labels)
        self.assertTrue(len(a)==0)
        self.assertTrue(len(p)==0)
        self.assertTrue(len(n)==0)

if __name__ == '__main__':
    unittest.main()