import unittest
import torch
import numpy as np
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.reducers import MeanReducer

class TestTripletMarginLoss(unittest.TestCase):
    def test_triplet_margin_loss(self):
        margin = 0.2
        loss_funcA = TripletMarginLoss(margin=margin)
        loss_funcB = TripletMarginLoss(margin=margin, reducer=MeanReducer())
        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])

        lossA = loss_funcA(embeddings, labels)
        lossB = loss_funcB(embeddings, labels)
        lossA.backward()
        lossB.backward()
        
        triplets = [(0,1,2), (0,1,3), (0,1,4), (1,0,2), (1,0,3), (1,0,4), (2,3,0), (2,3,1), (2,3,4), (3,2,0), (3,2,1), (3,2,4)]

        correct_loss = 0
        num_non_zero_triplets = 0
        for a, p, n in triplets:
            anchor, positive, negative = embeddings[a], embeddings[p], embeddings[n]
            curr_loss = torch.relu(torch.sqrt(torch.sum((anchor-positive)**2)) - torch.sqrt(torch.sum((anchor-negative)**2)) + margin)
            if curr_loss > 0:
                num_non_zero_triplets += 1
            correct_loss += curr_loss
        self.assertTrue(torch.isclose(lossA, correct_loss/num_non_zero_triplets))
        self.assertTrue(torch.isclose(lossB, correct_loss/len(triplets)))


    def test_with_no_valid_triplets(self):
        loss_funcA = TripletMarginLoss(margin=0.2)
        loss_funcB = TripletMarginLoss(margin=0.2, reducer=MeanReducer())
        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float) #2D embeddings
        labels = torch.LongTensor([0, 1, 2, 3, 4])
        lossA = loss_funcA(embeddings, labels)
        lossA.backward()
        lossB = loss_funcB(embeddings, labels)
        lossB.backward()
        self.assertEqual(lossA, 0)
        self.assertEqual(lossB, 0)