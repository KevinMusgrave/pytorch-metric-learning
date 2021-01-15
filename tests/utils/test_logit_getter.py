import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import (
    ArcFaceLoss,
    NormalizedSoftmaxLoss,
    ProxyAnchorLoss,
)
from pytorch_metric_learning.utils.inference import LogitGetter

from .. import TEST_DEVICE, TEST_DTYPES


class TestLogitGetter(unittest.TestCase):
    def helper_tester(self, loss_fn, embeddings, batch_size, num_classes, **kwargs):
        LG = LogitGetter(loss_fn, **kwargs)
        logits = LG(embeddings)
        self.assertTrue(logits.size() == torch.Size([batch_size, num_classes]))

    def test_logit_getter(self):
        embedding_size = 512
        num_classes = 10
        batch_size = 32

        for dtype in TEST_DTYPES:
            embeddings = (
                torch.randn(batch_size, embedding_size).to(TEST_DEVICE).type(dtype)
            )
            kwargs = {"num_classes": num_classes, "embedding_size": embedding_size}
            loss1 = ArcFaceLoss(**kwargs).to(TEST_DEVICE).type(dtype)
            loss2 = NormalizedSoftmaxLoss(**kwargs).to(TEST_DEVICE).type(dtype)
            loss3 = ProxyAnchorLoss(**kwargs).to(TEST_DEVICE).type(dtype)

            # test the ability to infer shape
            for loss in [loss1, loss2, loss3]:
                self.helper_tester(loss, embeddings, batch_size, num_classes)

            # test specifying wrong layer name
            self.assertRaises(AttributeError, LogitGetter, loss1, layer_name="blah")

            # test specifying correct layer name
            self.helper_tester(
                loss1, embeddings, batch_size, num_classes, layer_name="W"
            )

            # test specifying a distance metric
            self.helper_tester(
                loss1, embeddings, batch_size, num_classes, distance=LpDistance()
            )

            # test specifying transpose incorrectly
            LG = LogitGetter(loss1, transpose=False)
            self.assertRaises(RuntimeError, LG, embeddings)

            # test specifying transpose correctly
            self.helper_tester(
                loss1, embeddings, batch_size, num_classes, transpose=True
            )

            # test copying weights
            LG = LogitGetter(loss1)
            self.assertTrue(torch.all(LG.weights == loss1.W))
            loss1.W.data *= 0
            self.assertTrue(not torch.all(LG.weights == loss1.W))

            # test not copying weights
            LG = LogitGetter(loss1, copy_weights=False)
            self.assertTrue(torch.all(LG.weights == loss1.W))
            loss1.W.data *= 0
            self.assertTrue(torch.all(LG.weights == loss1.W))


if __name__ == "__main__":
    unittest.main()
