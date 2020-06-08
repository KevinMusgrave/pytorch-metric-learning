import unittest
import torch
from pytorch_metric_learning.losses import NPairsLoss
from pytorch_metric_learning.utils import common_functions as c_f, loss_and_miner_utils as lmu

class TestNPairsLoss(unittest.TestCase):
    def test_npairs_loss(self):
        loss_funcA = NPairsLoss()
        loss_funcB = NPairsLoss(l2_reg_weight=1)

        embedding_angles = list(range(0,180,20))[:7]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 1, 2, 3])

        lossA = loss_funcA(embeddings, labels)
        lossB = loss_funcB(embeddings, labels)
        pos_pairs = [(0,1), (2,3)]
        neg_pairs = [(0,3), (2,1)]

        total_loss = 0
        for a1, p in pos_pairs:
            anchor, positive = embeddings[a1], embeddings[p]
            numerator = torch.exp(torch.matmul(anchor, positive))
            denominator = numerator.clone()
            for a2, n in neg_pairs:
                if a2 == a1:
                    negative = embeddings[n]
                    denominator += torch.exp(torch.matmul(anchor, negative))
            curr_loss = -torch.log(numerator/denominator)
            total_loss += curr_loss
        
        total_loss /= len(pos_pairs[0])
        self.assertTrue(torch.isclose(lossA, total_loss))
        self.assertTrue(torch.isclose(lossB, total_loss+1)) # l2_reg is going to be 1 since the embeddings are normalized


    def test_with_no_valid_pairs(self):
        loss_func = NPairsLoss()
        embedding_angles = [0]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0])
        self.assertEqual(loss_func(embeddings, labels), 0)