import unittest
import torch
import numpy as np
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.utils import common_functions as c_f, loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import MeanReducer

class TestContrastiveLoss(unittest.TestCase):
    def test_contrastive_loss(self):
        loss_funcA = ContrastiveLoss(pos_margin=0.25, neg_margin=1.5, use_similarity=False, squared_distances=True)
        loss_funcB = ContrastiveLoss(pos_margin=1.5, neg_margin=0.6, use_similarity=True)
        loss_funcC = ContrastiveLoss(pos_margin=0.25, neg_margin=1.5, use_similarity=False, squared_distances=True, reducer=MeanReducer())
        loss_funcD = ContrastiveLoss(pos_margin=1.5, neg_margin=0.6, use_similarity=True, reducer=MeanReducer())

        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])

        lossA = loss_funcA(embeddings, labels)
        lossB = loss_funcB(embeddings, labels)
        lossC = loss_funcC(embeddings, labels)
        lossD = loss_funcD(embeddings, labels)

        pos_pairs = [(0,1), (1,0), (2,3), (3,2)]
        neg_pairs = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,4), (3,0), (3,1), (3,4), (4,0), (4,1), (4,2), (4,3)]

        correct_pos_losses = [0,0,0,0]
        correct_neg_losses = [0,0,0,0]
        num_non_zero_pos = [0,0,0,0]
        num_non_zero_neg = [0,0,0,0]
        for a,p in pos_pairs:
            anchor, positive = embeddings[a], embeddings[p]
            correct_lossA = torch.relu(torch.sum((anchor-positive)**2) - 0.25)
            correct_lossB = torch.relu(1.5-torch.matmul(anchor, positive))
            correct_pos_losses[0] += correct_lossA
            correct_pos_losses[1] += correct_lossB
            correct_pos_losses[2] += correct_lossA
            correct_pos_losses[3] += correct_lossB
            if correct_lossA > 0:
                num_non_zero_pos[0] += 1
                num_non_zero_pos[2] += 1
            if correct_lossB > 0:
                num_non_zero_pos[1] += 1
                num_non_zero_pos[3] += 1          

        for a,n in neg_pairs:
            anchor, negative = embeddings[a], embeddings[n]
            correct_lossA = torch.relu(1.5 - torch.sum((anchor-negative)**2))
            correct_lossB = torch.relu(torch.matmul(anchor, negative)-0.6)
            correct_neg_losses[0] += correct_lossA
            correct_neg_losses[1] += correct_lossB
            correct_neg_losses[2] += correct_lossA
            correct_neg_losses[3] += correct_lossB
            if correct_lossA > 0:
                num_non_zero_neg[0] += 1
                num_non_zero_neg[2] += 1
            if correct_lossB > 0:
                num_non_zero_neg[1] += 1
                num_non_zero_neg[3] += 1
                
        for i in range(2):
            if num_non_zero_pos[i] > 0:
                correct_pos_losses[i] /= num_non_zero_pos[i]
            if num_non_zero_neg[i] > 0:
                correct_neg_losses[i] /= num_non_zero_neg[i]
        
        for i in range(2,4):
            correct_pos_losses[i] /= len(pos_pairs)
            correct_neg_losses[i] /= len(neg_pairs)

        correct_losses = [0,0,0,0]
        for i in range(4):
            correct_losses[i] = correct_pos_losses[i] + correct_neg_losses[i]

        self.assertTrue(torch.isclose(lossA, correct_losses[0]))
        self.assertTrue(torch.isclose(lossB, correct_losses[1]))
        self.assertTrue(torch.isclose(lossC, correct_losses[2]))
        self.assertTrue(torch.isclose(lossD, correct_losses[3]))


    def test_with_no_valid_pairs(self):
        lossA = ContrastiveLoss(use_similarity=False)
        lossB = ContrastiveLoss(use_similarity=True)
        embedding_angles = [0]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0])
        self.assertEqual(lossA(embeddings, labels), 0)
        self.assertEqual(lossB(embeddings, labels), 0)