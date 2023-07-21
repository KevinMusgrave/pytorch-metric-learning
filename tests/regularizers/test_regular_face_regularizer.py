import unittest

import torch

from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.regularizers import RegularFaceRegularizer
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestRegularFaceRegularizer(unittest.TestCase):
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
                regularizer=RegularFaceRegularizer(),
                reg_weight=reg_weight,
            ).to(TEST_DEVICE)

            embeddings = torch.nn.functional.normalize(
                torch.randn((180, embedding_size), requires_grad=True)
                .type(dtype)
                .to(TEST_DEVICE)
            )
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

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
