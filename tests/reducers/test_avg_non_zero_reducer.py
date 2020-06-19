import unittest
import torch
from pytorch_metric_learning.reducers import AvgNonZeroReducer

class TestAvgNonZeroReducer(unittest.TestCase):
    def test_avg_non_zero_reducer(self):
        reducer = AvgNonZeroReducer()
        batch_size = 100
        embedding_size = 64
        embeddings = torch.randn(batch_size, embedding_size)
        labels = torch.randint(0,10,(batch_size,))
        pair_indices = (torch.randint(0,batch_size,(batch_size,)), torch.randint(0,batch_size,(batch_size,)))
        triplet_indices = pair_indices + (torch.randint(0,batch_size,(batch_size,)),)
        losses = torch.randn(batch_size)
        zero_losses = torch.zeros(batch_size)

        for indices, reduction_type in [(torch.arange(batch_size), "element"),
                                        (pair_indices, "pos_pair"),
                                        (pair_indices, "neg_pair"),
                                        (triplet_indices, "triplet")]:
            for L in [losses, zero_losses]:
                loss_dict = {"loss": {"losses": L, "indices": indices, "reduction_type": reduction_type}}
                output = reducer(loss_dict, embeddings, labels)
                filtered_L = L[L>0]
                if len(filtered_L) > 0:
                    correct_output = torch.mean(filtered_L)
                else:
                    correct_output = torch.mean(L)*0
                self.assertTrue(output == correct_output)