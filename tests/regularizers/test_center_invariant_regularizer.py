import unittest
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.regularizers import CenterInvariantRegularizer


class TestCenterInvariantRegularizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device("cuda")

    def test_center_invariant_regularizer(self):
        temperature = 0.1
        num_classes = 10
        embedding_size = 512
        reg_weight = 0.1
        for dtype in TEST_DTYPES:
            loss_func = NormalizedSoftmaxLoss(
                temperature=temperature,
                num_classes=num_classes,
                embedding_size=embedding_size,
                weight_regularizer=CenterInvariantRegularizer(),
                weight_reg_weight=reg_weight,
            ).to(self.device)

            embeddings = torch.nn.functional.normalize(
                torch.randn((180, embedding_size), requires_grad=True)
                .type(dtype)
                .to(self.device)
            )
            labels = torch.randint(low=0, high=10, size=(180,)).to(self.device)

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
            logits = torch.matmul(embeddings, weights)
            correct_class_loss = torch.nn.functional.cross_entropy(
                logits / temperature, labels
            )

            correct_reg_loss = 0
            average_squared_weight_norms = 0
            for i in range(num_classes):
                average_squared_weight_norms += torch.norm(loss_func.W[:, i], p=2) ** 2
            average_squared_weight_norms /= num_classes
            for i in range(num_classes):
                deviation = (
                    torch.norm(loss_func.W[:, i], p=2) ** 2
                    - average_squared_weight_norms
                )
                correct_reg_loss += (deviation ** 2) / 4
            correct_reg_loss /= num_classes

            correct_total_loss = correct_class_loss + (correct_reg_loss * reg_weight)
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total_loss, rtol=rtol))
