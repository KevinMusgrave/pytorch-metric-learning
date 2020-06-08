import unittest
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestNormalizedSoftmaxLoss(unittest.TestCase):
    def test_normalized_softmax_loss(self):
        temperature = 0.1
        loss_func = NormalizedSoftmaxLoss(temperature=temperature, num_classes=10, embedding_size=2)

        embedding_angles = [0, 20, 40, 60, 80]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
        labels = torch.LongTensor([0, 0, 1, 1, 2])

        loss = loss_func(embeddings, labels)

        weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
        logits = torch.matmul(embeddings, weights)
        correct_loss = torch.nn.functional.cross_entropy(logits/temperature, labels)
        self.assertTrue(torch.isclose(loss, correct_loss))