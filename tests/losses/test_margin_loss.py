import unittest
import torch
import numpy as np
from pytorch_metric_learning.losses import MarginLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestMarginLoss(unittest.TestCase):
    def test_margin_loss(self):
        margin, nu, beta = 0.1, 0, 1
        loss_func = MarginLoss(margin=margin, nu=nu, beta=beta)

        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])

        loss = loss_func(embeddings, labels)

        triplets = [(0,1,2), (0,1,3), (0,1,4), (1,0,2), (1,0,3), (1,0,4), (2,3,0), (2,3,1), (2,3,4), (3,2,0), (3,2,1), (3,2,4)]

        correct_total_loss = 0
        num_non_zero = 0
        for a, p, n in triplets:
            anchor, positive, negative = embeddings[a], embeddings[p], embeddings[n]
            pos_loss = torch.relu(torch.sqrt(torch.sum((anchor-positive)**2)) - beta + margin)
            neg_loss = torch.relu(beta - torch.sqrt(torch.sum((anchor-negative)**2)) + margin)
            correct_total_loss += pos_loss + neg_loss
            if pos_loss > 0:
                num_non_zero += 1
            if neg_loss > 0:
                num_non_zero += 1
                
        if num_non_zero > 0:
            correct_total_loss /= num_non_zero

        self.assertTrue(torch.isclose(loss, correct_total_loss))


    def test_with_no_valid_triplets(self):
        margin, nu, beta = 0.1, 0, 1
        loss_func = MarginLoss(margin=margin, nu=nu, beta=beta)
        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0, 1, 2, 3, 4])
        self.assertEqual(loss_func(embeddings, labels), 0)