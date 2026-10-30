import unittest

import torch

from pytorch_metric_learning.losses import TripletMarginLoss


class TestCommonFunctions(unittest.TestCase):
    def test_check_shapes(self):
        three_dimensional_embeddings = torch.randn(32, 512, 3)
        labels = torch.randint(0, 10, size=(32,))
        loss_fn = TripletMarginLoss()
        self.assertRaises(
            ValueError, lambda: loss_fn(three_dimensional_embeddings, labels)
        )

        mismatched_embeddings = torch.randn(33, 512)
        self.assertRaises(ValueError, lambda: loss_fn(mismatched_embeddings, labels))

        embeddings = torch.randn(32, 512)
        two_dimensional_labels = labels.unsqueeze(1)
        self.assertRaises(
            ValueError, lambda: loss_fn(embeddings, two_dimensional_labels)
        )

        self.assertTrue(torch.is_tensor(loss_fn(embeddings, labels)))


if __name__ == "__main__":
    unittest.main()
