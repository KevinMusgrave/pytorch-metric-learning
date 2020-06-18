import unittest
import torch
from pytorch_metric_learning.reducers import ThresholdReducer

class TestThresholdReducer(unittest.TestCase):
    def test_threshold_reducer(self):
        threshold = 0.5
        reducer = ThresholdReducer(threshold)
        batch_size = 100
        embedding_size = 64
        embeddings = torch.randn(batch_size, embedding_size)
        labels = torch.randint(0,10,(batch_size,))
        pair_indices = (torch.randint(0,10,(batch_size,)), torch.randint(0,10,(batch_size,)))
        triplet_indices = pair_indices + (torch.randint(0,10,(batch_size,)),)
        losses = torch.randn(batch_size)

        for indices, reduction_type in [(torch.arange(batch_size), "element"),
                                        (pair_indices, "pos_pair"),
                                        (pair_indices, "neg_pair"),
                                        (triplet_indices, "triplet")]:
            loss_dict = {"loss": {"losses": losses, "indices": indices, "reduction_type": reduction_type}}
            output = reducer(loss_dict, embeddings, labels)
            correct_output = torch.mean(losses[losses>threshold])
            self.assertTrue(output == correct_output)