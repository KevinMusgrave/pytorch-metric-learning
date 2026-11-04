import unittest

import pytest
import torch

from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.regularizers import CenterInvariantRegularizer

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TestCenterInvariantRegularizer(unittest.TestCase):
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

            correct_reg_loss = 0
            average_squared_weight_norms = 0
            for i in range(num_classes):
                average_squared_weight_norms += (
                    torch.norm(loss_func.weight[i].double(), p=2) ** 2
                )
            average_squared_weight_norms /= num_classes
            for i in range(num_classes):
                deviation = (
                    torch.norm(loss_func.weight[i].double(), p=2) ** 2
                    - average_squared_weight_norms
                )
                correct_reg_loss += (deviation**2) / 4
            correct_reg_loss /= num_classes

            correct_total_loss = correct_class_loss + (correct_reg_loss * reg_weight)
            helpers.assert_close(loss, correct_total_loss, dtype)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize(
    "data",
    [
        [[16.0, 0.0], [17.0, 0.0]],
        [[300.0, 0.0], [301.0, 0.0]],
        [[2.0, 3.0], [4.0, 5.0]],
    ],
)
def test_center_invariant_matches_squared_norm_variance(dtype, data):
    centers = torch.tensor(data, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    reference = centers.detach().double()
    squared_norms = reference.square().sum(dim=1)
    deviations = squared_norms - squared_norms.mean()
    expected = deviations.square().mean() / 4
    expected_gradient = deviations[:, None] * reference / len(data)

    loss = CenterInvariantRegularizer()(centers)

    expected_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    assert loss.dtype == expected_dtype
    torch.testing.assert_close(loss, expected.to(expected_dtype))
    loss.backward()
    torch.testing.assert_close(centers.grad, expected_gradient.to(dtype))
