import unittest
import torch
from pytorch_metric_learning.reducers import MeanReducer

class TestMeanReducer(unittest.TestCase):
    def test_mean_reducer(self):
        reducer = MeanReducer()
        batch_size = 100
        embedding_size = 64
        embeddings = torch.randn(batch_size, embedding_size)
        labels = torch.randint(0,10,(batch_size,))
        pair_indices = (torch.randint(0,batch_size,(batch_size,)), torch.randint(0,batch_size,(batch_size,)))
        triplet_indices = pair_indices + (torch.randint(0,batch_size,(batch_size,)),)
        losses = torch.randn(batch_size)

        for indices, reduction_type in [(torch.arange(batch_size), "element"),
                                        (pair_indices, "pos_pair"),
                                        (pair_indices, "neg_pair"),
                                        (triplet_indices, "triplet")]:
            loss_dict = {"loss": {"losses": losses, "indices": indices, "reduction_type": reduction_type}}
            output = reducer(loss_dict, embeddings, labels)
            correct_output = torch.mean(losses)
            self.assertTrue(output == correct_output)