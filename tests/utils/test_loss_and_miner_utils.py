import unittest
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch

class TestLossAndMinerUtils(unittest.TestCase):

    def test_logsumexp(self):
        mat = torch.FloatTensor([[-1, 0, 1, 10, 50],
                                [-30, -20, 0, 20, 30],
                                [10, 20, 30, 40, 50],
                                [0,0,0,0,0]])
        result = lmu.logsumexp(mat, keep_mask=None, add_one=False, dim=1)
        correct_result = torch.logsumexp(mat, dim=1, keepdim=True)
        self.assertTrue(torch.equal(result, correct_result))

        result = lmu.logsumexp(mat, keep_mask=None, add_one=True, dim=1)
        correct_result = torch.logsumexp(torch.cat([mat, torch.zeros(mat.size(0)).unsqueeze(1)], dim=1), dim=1, keepdim=True)
        self.assertTrue(torch.equal(result, correct_result))

        keep_mask = torch.FloatTensor([[1, 1, 0, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 1, 1, 0, 0]])
        result = lmu.logsumexp(mat, keep_mask=keep_mask, add_one=False, dim=1)

        row0 = torch.log(torch.sum(torch.exp(torch.FloatTensor([-1, 0])))).unsqueeze(0)
        row1 = torch.log(torch.sum(torch.exp(torch.FloatTensor([-20, 0, 20])))).unsqueeze(0)
        row2 = torch.FloatTensor([0.])
        row3 = torch.log(torch.sum(torch.exp(torch.FloatTensor([0, 0])))).unsqueeze(0)
        correct_result = torch.stack([row0, row1, row2, row3], dim=0)
        self.assertTrue(torch.allclose(result, correct_result))


    def test_get_all_pairs_triplets_indices(self):
        original_x = torch.arange(10)

        for i in range(1, 11):
            x = original_x.repeat(i)
            correct_num_pos = len(x)*(i-1)
            correct_num_neg = len(x)*(len(x)-i)
            a1, p, a2, n = lmu.get_all_pairs_indices(x)
            self.assertTrue(len(a1) == len(p) == correct_num_pos)
            self.assertTrue(len(a2) == len(n) == correct_num_neg)

            correct_num_triplets = len(x)*(i-1)*(len(x)-i)
            a, p, n = lmu.get_all_triplets_indices(x)
            self.assertTrue(len(a) == len(p) == len(n) == correct_num_triplets)

            
if __name__ == '__main__':
    unittest.main()