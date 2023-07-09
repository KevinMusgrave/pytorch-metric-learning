import unittest

import torch

from pytorch_metric_learning.reducers import ClassWeightedReducer

from .. import TEST_DEVICE, TEST_DTYPES, WITH_COLLECT_STATS


class TestClassWeightedReducer(unittest.TestCase):
    def test_class_weighted_reducer_with_threshold(self):
        torch.manual_seed(99115)
        class_weights = torch.tensor([1, 0.9, 1, 0.1, 0, 0, 0, 0, 0, 0])
        for low_threshold, high_threshold in [
            (None, None),
            (0.1, None),
            (None, 0.2),
            (0.1, 0.2),
        ]:
            reducer = ClassWeightedReducer(
                class_weights, low=low_threshold, high=high_threshold
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
                losses = torch.randn(batch_size).type(dtype).to(TEST_DEVICE)
                zero_losses = torch.zeros(batch_size).type(dtype).to(TEST_DEVICE)

                for indices, reduction_type in [
                    (torch.arange(batch_size), "element"),
                    (pair_indices, "pos_pair"),
                    (pair_indices, "neg_pair"),
                    (triplet_indices, "triplet"),
                ]:
                    for L in [losses, zero_losses]:
                        loss_dict = {
                            "loss": {
                                "losses": L,
                                "indices": indices,
                                "reduction_type": reduction_type,
                            }
                        }
                        output = reducer(loss_dict, embeddings, labels)
                        if low_threshold is not None:
                            L = L[L > low_threshold]
                        if high_threshold is not None:
                            L = L[L < high_threshold]
                        if len(L) > 0:
                            correct_output = 0
                            for i in range(len(L)):
                                if reduction_type == "element":
                                    batch_idx = indices[i]
                                else:
                                    batch_idx = indices[0][i]
                                class_label = labels[batch_idx]
                                correct_output += (
                                    L[i]
                                    * class_weights.type(dtype).to(TEST_DEVICE)[
                                        class_label
                                    ]
                                )
                            correct_output /= len(L)
                        else:
                            correct_output = 0
                        rtol = 1e-2 if dtype == torch.float16 else 1e-5
                        self.assertTrue(
                            torch.isclose(output, correct_output, rtol=rtol)
                        )

                        if WITH_COLLECT_STATS:
                            self.assertTrue(reducer.num_past_filter == len(L))
