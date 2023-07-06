import unittest

import torch

from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    DivisorReducer,
    MultipleReducers,
)

from .. import TEST_DEVICE, TEST_DTYPES


class TestMultipleReducers(unittest.TestCase):
    def test_multiple_reducers(self):
        reducer = MultipleReducers(
            {"lossA": AvgNonZeroReducer(), "lossB": DivisorReducer()}
        )
        batch_size = 100
        embedding_size = 64
        for dtype in TEST_DTYPES:
            embeddings = (
                torch.randn(batch_size, embedding_size).type(dtype).to(TEST_DEVICE)
            )
            labels = torch.randint(0, 10, (batch_size,))
            pair_indices = (
                torch.randint(0, batch_size, (batch_size,)),
                torch.randint(0, batch_size, (batch_size,)),
            )
            triplet_indices = pair_indices + (
                torch.randint(0, batch_size, (batch_size,)),
            )
            lossesA = torch.randn(batch_size).type(dtype).to(TEST_DEVICE)
            lossesB = torch.randn(batch_size).type(dtype).to(TEST_DEVICE)

            for indices, reduction_type in [
                (torch.arange(batch_size), "element"),
                (pair_indices, "pos_pair"),
                (pair_indices, "neg_pair"),
                (triplet_indices, "triplet"),
            ]:
                loss_dict = {
                    "lossA": {
                        "losses": lossesA,
                        "indices": indices,
                        "reduction_type": reduction_type,
                    },
                    "lossB": {
                        "losses": lossesB,
                        "indices": indices,
                        "reduction_type": reduction_type,
                        "divisor": 32 + 15,
                    },
                }
                output = reducer(loss_dict, embeddings, labels)
                correct_output = (torch.mean(lossesA[lossesA > 0])) + (
                    torch.sum(lossesB) / (32 + 15)
                )
                rtol = 1e-2 if dtype == torch.float16 else 1e-5
                self.assertTrue(torch.isclose(output, correct_output, rtol=rtol))
