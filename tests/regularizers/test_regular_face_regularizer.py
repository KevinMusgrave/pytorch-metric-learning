import unittest

import pytest
import torch

from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.regularizers import RegularFaceRegularizer
from pytorch_metric_learning.utils import _common as c_f

from .. import TEST_DEVICE, TEST_DTYPES, helpers


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
                weight_regularizer=RegularFaceRegularizer(),
                weight_reg_weight=reg_weight,
            ).to(TEST_DEVICE, dtype)

            embeddings = torch.nn.functional.normalize(
                torch.randn((180, embedding_size), requires_grad=True)
                .type(dtype)
                .to(TEST_DEVICE)
            )
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = torch.nn.functional.normalize(loss_func.weight, p=2, dim=1).t()
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
            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(loss, correct_total_loss, rtol=rtol))

    def test_distance_must_be_a_similarity(self):
        with self.assertRaises(TypeError):
            RegularFaceRegularizer(distance=LpDistance())


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("size", [0, 1])
@pytest.mark.parametrize("distance_type", [CosineSimilarity, DotProductSimilarity])
def test_regular_face_without_other_centers_has_zero_loss(dtype, size, distance_type):
    centers = torch.tensor([[2.0, 3.0]], dtype=dtype, device=TEST_DEVICE)[
        :size
    ].requires_grad_()
    result = RegularFaceRegularizer(distance=distance_type())(centers)
    torch.testing.assert_close(result, centers.new_zeros(()))
    result.backward()
    torch.testing.assert_close(centers.grad, torch.zeros_like(centers))


@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_regular_face_two_centers_keeps_negative_similarity(dtype):
    centers = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0]], dtype=dtype, device=TEST_DEVICE, requires_grad=True
    )
    result = RegularFaceRegularizer()(centers)
    torch.testing.assert_close(result, centers.new_tensor(-1.0))
    result.backward()
    torch.testing.assert_close(centers.grad, torch.zeros_like(centers))
