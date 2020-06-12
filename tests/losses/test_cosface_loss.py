import unittest
import torch
import numpy as np
from pytorch_metric_learning.losses import CosFaceLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestCosFaceLoss(unittest.TestCase):
    def test_cosface_loss(self):
        margin = 0.5
        scale = 64
        loss_func = CosFaceLoss(margin=margin, scale=scale, num_classes=10, embedding_size=2)

        embedding_angles = torch.arange(0, 180)
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.randint(low=0, high=10, size=(180,))

        loss = loss_func(embeddings, labels)

        weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
        logits = torch.matmul(embeddings, weights)
        for i, c in enumerate(labels):
            logits[i, c] -= margin
        
        correct_loss = torch.nn.functional.cross_entropy(logits*scale, labels)
        self.assertTrue(torch.isclose(loss, correct_loss))