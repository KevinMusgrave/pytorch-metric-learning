import unittest

import torch

from pytorch_metric_learning.reducers import DivisorReducer

from .. import TEST_DEVICE, TEST_DTYPES


class TestDivisorReducer(unittest.TestCase):
    def test_divisor_reducer(self):
        reducer = DivisorReducer()
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
            losses = torch.randn(batch_size).type(dtype).to(TEST_DEVICE)

            for indices, reduction_type in [
                (torch.arange(batch_size), "element"),
                (pair_indices, "pos_pair"),
                (pair_indices, "neg_pair"),
                (triplet_indices, "triplet"),
                (None, "already_reduced"),
            ]:
                for partA, partB in [(0, 0), (32, 15)]:
                    if reduction_type == "already_reduced":
                        L = losses[0]
                    else:
                        L = losses
                    loss_dict = {
                        "loss": {
                            "losses": L,
                            "indices": indices,
                            "reduction_type": reduction_type,
                            "divisor": partA + partB,
                        }
                    }
                    output = reducer(loss_dict, embeddings, labels)
                    if reduction_type == "already_reduced":
                        correct_output = L
                    elif partA + partB == 0:
                        correct_output = torch.sum(L) * 0
                    else:
                        correct_output = torch.sum(L) / (32 + 15)
                    self.assertTrue(output == correct_output)

            loss_dict = {
                "loss": {
                    "losses": losses[0],
                    "indices": None,
                    "reduction_type": "already_reduced",
                }
            }
            output = reducer(loss_dict, embeddings, labels)
            correct_output = losses[0]
            self.assertTrue(output == correct_output)
