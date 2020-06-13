import unittest
import torch
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestNTXentLoss(unittest.TestCase):
    def test_ntxent_loss(self):
        temperature = 0.1
        loss_func = NTXentLoss(temperature=temperature)

        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])

        loss = loss_func(embeddings, labels)
        loss.backward()

        pos_pairs = [(0,1), (1,0), (2,3), (3,2)]
        neg_pairs = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,4), (3,0), (3,1), (3,4), (4,0), (4,1), (4,2), (4,3)]

        total_loss = 0
        for a1,p in pos_pairs:
            anchor, positive = embeddings[a1], embeddings[p]
            numerator = torch.exp(torch.matmul(anchor, positive)/temperature)
            denominator = numerator.clone()
            for a2,n in neg_pairs:
                if a2 == a1:
                    negative = embeddings[n]
                else:
                    continue
                denominator += torch.exp(torch.matmul(anchor, negative)/temperature)
            curr_loss = -torch.log(numerator/denominator)
            total_loss += curr_loss
        
        total_loss /= len(pos_pairs)
        self.assertTrue(torch.isclose(loss, total_loss))


    def test_with_no_valid_pairs(self):
        loss = NTXentLoss(temperature=0.1)
        embedding_angles = [0]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0])
        self.assertEqual(loss(embeddings, labels), 0)