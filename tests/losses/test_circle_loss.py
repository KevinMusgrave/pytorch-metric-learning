import unittest
import torch
from pytorch_metric_learning.losses import CircleLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestCircleLoss(unittest.TestCase):
    def test_circle_loss(self):
        margin, gamma = 0.4, 2
        Op, On = 1+margin, -margin
        delta_p, delta_n = 1-margin, margin
        loss_func = CircleLoss(m=margin, gamma=gamma)

        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])

        loss = loss_func(embeddings, labels)
        loss.backward()

        pos_pairs = [(0,1), (1,0), (2,3), (3,2)]
        neg_pairs = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,4), (3,0), (3,1), (3,4), (4,0), (4,1), (4,2), (4,3)]

        correct_total = 0
        totals = []
        for i in range(len(embeddings)):
            pos_exp = 0
            neg_exp = 0
            for a,p in pos_pairs:
                if a == i:
                    anchor, positive = embeddings[a], embeddings[p]
                    ap_sim = torch.matmul(anchor,positive)
                    logit_p = -gamma*torch.relu(Op-ap_sim)*(ap_sim-delta_p)
                    pos_exp += torch.exp(logit_p)

            for a,n in neg_pairs:
                if a == i:
                    anchor, negative = embeddings[a], embeddings[n]
                    an_sim = torch.matmul(anchor,negative)
                    logit_n = gamma*torch.relu(an_sim-On)*(an_sim-delta_n)
                    neg_exp += torch.exp(logit_n)

            totals.append(torch.log(1+pos_exp*neg_exp))
            correct_total += torch.log(1+pos_exp*neg_exp)

        correct_total /= 4 # only 4 of the embeddings have both pos and neg pairs
        self.assertTrue(torch.isclose(loss, correct_total))


    def test_with_no_valid_pairs(self):
        margin, gamma = 0.4, 80
        loss_func = CircleLoss(m=margin, gamma=gamma)
        embedding_angles = [0]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0])
        self.assertEqual(loss_func(embeddings, labels), 0)
    
    def test_overflow(self):
        margin, gamma = 0.4, 300
        loss_func = CircleLoss(m=margin, gamma=gamma)
        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])
        loss = loss_func(embeddings, labels)
        loss.backward()
        self.assertTrue(not torch.isnan(loss) and not torch.isinf(loss))