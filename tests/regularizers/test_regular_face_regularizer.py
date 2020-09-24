import unittest
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.regularizers import RegularFaceRegularizer
from pytorch_metric_learning.utils import common_functions as c_f


class TestRegularFaceRegularizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device("cuda")

    def test_regular_face_regularizer(self):
        temperature = 0.1
        num_classes = 10
        embedding_size = 512
        reg_weight = 0.1
        for dtype in TEST_DTYPES:
            loss_func = NormalizedSoftmaxLoss(
                temperature=temperature,
                num_classes=num_classes,
                embedding_size=embedding_size,
                weight_regularizer=RegularFaceRegularizer(),
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

            weight_cos_matrix = torch.matmul(weights.t(), weights)
            weight_cos_matrix.fill_diagonal_(c_f.neg_inf(dtype))
            correct_reg_loss = 0
            for i in range(num_classes):
                correct_reg_loss += torch.max(weight_cos_matrix[i])
            correct_reg_loss /= num_classes

            correct_total_loss = correct_class_loss + (correct_reg_loss * reg_weight)
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total_loss, rtol=rtol))
