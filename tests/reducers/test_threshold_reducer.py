import unittest 
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.reducers import ThresholdReducer

class TestThresholdReducer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_threshold_reducer(self):
        threshold = 0.5
        reducer = ThresholdReducer(threshold)
        batch_size = 100
        embedding_size = 64
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, embedding_size).type(dtype).to(self.device)
            labels = torch.randint(0,10,(batch_size,))
            pair_indices = (torch.randint(0,batch_size,(batch_size,)), torch.randint(0,batch_size,(batch_size,)))
            triplet_indices = pair_indices + (torch.randint(0,batch_size,(batch_size,)),)
            losses = torch.randn(batch_size).type(dtype).to(self.device)
            zero_losses = torch.zeros(batch_size).type(dtype).to(self.device)

            for indices, reduction_type in [(torch.arange(batch_size), "element"),
                                            (pair_indices, "pos_pair"),
                                            (pair_indices, "neg_pair"),
                                            (triplet_indices, "triplet")]:
                for L in [losses, zero_losses]:
                    loss_dict = {"loss": {"losses": L, "indices": indices, "reduction_type": reduction_type}}
                    output = reducer(loss_dict, embeddings, labels)
                    filtered_L = L[L>threshold]
                    if len(filtered_L) > threshold:
                        correct_output = torch.mean(filtered_L)
                    else:
                        correct_output = torch.mean(L)*0
                    self.assertTrue(output == correct_output)