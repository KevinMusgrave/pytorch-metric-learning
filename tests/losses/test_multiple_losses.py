import unittest
from .. import TEST_DTYPES, TEST_DEVICE
import torch
from pytorch_metric_learning.losses import (
    MultipleLosses,
    ContrastiveLoss,
    TripletMarginLoss,
)
from pytorch_metric_learning.utils import common_functions as c_f


class TestMultipleLosses(unittest.TestCase):
    def test_multiple_losses(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        loss_func = MultipleLosses(
            losses={"lossA": lossA, "lossB": lossB}, weights={"lossA": 1, "lossB": 0.23}
        )

        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            loss.backward()

            correct_loss = lossA(embeddings, labels) + lossB(embeddings, labels) * 0.23
            self.assertTrue(torch.isclose(loss, correct_loss))
