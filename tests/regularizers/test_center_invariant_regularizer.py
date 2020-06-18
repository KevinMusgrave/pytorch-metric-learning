import unittest
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.regularizers import CenterInvariantRegularizer

class TestCenterInvariantRegularizer(unittest.TestCase):
    def test_center_invariant_regularizer(self):
        temperature = 0.1
        num_classes = 10
        embedding_size = 512
        loss_func = NormalizedSoftmaxLoss(temperature=temperature, 
                                            num_classes=num_classes, 
                                            embedding_size=embedding_size,
                                            regularizer=CenterInvariantRegularizer())

        embeddings = torch.nn.functional.normalize(torch.randn((180, embedding_size), requires_grad=True, dtype=torch.float))
        labels = torch.randint(low=0, high=10, size=(180,))

        loss = loss_func(embeddings, labels)
        loss.backward()

        weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
        logits = torch.matmul(embeddings, weights)
        correct_class_loss = torch.nn.functional.cross_entropy(logits/temperature, labels)

        squared_weight_norms = torch.norm(loss_func.W, p=2, dim=0)**2
        deviations_from_mean = squared_weight_norms - torch.mean(squared_weight_norms)
        correct_reg_loss = torch.mean((deviations_from_mean**2) / 4)

        correct_total_loss = correct_class_loss+correct_reg_loss
        self.assertTrue(torch.isclose(loss, correct_total_loss))