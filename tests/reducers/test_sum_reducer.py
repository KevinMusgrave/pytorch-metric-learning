import unittest

import torch

from pytorch_metric_learning.reducers import SumReducer

from .. import TEST_DEVICE, TEST_DTYPES, WITH_COLLECT_STATS


class TestSumReducer(unittest.TestCase):
    def test_sum_reducer_with_thresholds(self):
        torch.manual_seed(99115)
        for low_threshold, high_threshold in [
            (None, None),
            (0.1, None),
            (None, 0.2),
            (0.1, 0.2),
        ]:
            reducer = SumReducer(low=low_threshold, high=high_threshold)
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
                            correct_output = torch.sum(L, dtype=dtype)
                        else:
                            correct_output = torch.zeros(
                                1, dtype=dtype, device=TEST_DEVICE
                            )
                        rtol = 1e-2 if dtype == torch.float16 else 1e-5
                        self.assertTrue(
                            torch.isclose(output, correct_output, rtol=rtol)
                        )

                        if WITH_COLLECT_STATS:
                            self.assertTrue(reducer.num_past_filter == len(L))
