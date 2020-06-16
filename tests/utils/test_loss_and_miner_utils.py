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


    def test_convert_to_triplets(self):
        a1 = torch.LongTensor([0,1,2,3])
        p = torch.LongTensor([4,4,4,4])
        a2 = torch.LongTensor([4,5,6,7])
        n = torch.LongTensor([5,5,6,6])
        triplets = lmu.convert_to_triplets((a1,p,a2,n), labels=torch.arange(7))
        self.assertTrue(all(len(x)==0 for x in triplets))

        a2 = torch.LongTensor([0,4,5,6])
        triplets = lmu.convert_to_triplets((a1,p,a2,n), labels=torch.arange(7))
        self.assertTrue(triplets==[torch.LongTensor([0]),torch.LongTensor([4]), torch.LongTensor([5])])

    def test_convert_to_weights(self):
        a = torch.LongTensor([0,1,2,3])
        p = torch.LongTensor([4,4,4,4])
        n = torch.LongTensor([5,5,6,6])
        weights = lmu.convert_to_weights((a,p,n), labels=torch.arange(7))
        correct_weights = torch.FloatTensor([0.25,0.25,0.25,0.25,1,0.5,0.5])
        self.assertTrue(torch.all(weights==correct_weights))
