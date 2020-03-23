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

if __name__ == '__main__':
    unittest.main()