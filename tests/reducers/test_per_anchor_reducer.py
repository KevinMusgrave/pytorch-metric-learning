import unittest

import torch

from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    MeanReducer,
    PerAnchorReducer,
)
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES


class TestPerAnchorReducer(unittest.TestCase):
    def test_per_anchor_reducer(self):
        for inner_reducer in [MeanReducer(), AvgNonZeroReducer()]:
            reducer = PerAnchorReducer(inner_reducer)
            batch_size = 10
            embedding_size = 64
            for dtype in TEST_DTYPES:
                embeddings = (
                    torch.randn(batch_size, embedding_size).type(dtype).to(TEST_DEVICE)
                )
                labels = torch.randint(0, 10, (batch_size,))
                pos_pair_indices = lmu.get_all_pairs_indices(labels)[:2]
                neg_pair_indices = lmu.get_all_pairs_indices(labels)[2:]
                triplet_indices = lmu.get_all_triplets_indices(labels)

                for indices, reduction_type in [
                    (torch.arange(batch_size), "element"),
                    (pos_pair_indices, "pos_pair"),
                    (neg_pair_indices, "neg_pair"),
                    (triplet_indices, "triplet"),
                ]:
                    loss_size = (
                        len(indices) if reduction_type == "element" else len(indices[0])
                    )
                    losses = torch.randn(loss_size).type(dtype).to(TEST_DEVICE)
                    loss_dict = {
                        "loss": {
                            "losses": losses,
                            "indices": indices,
                            "reduction_type": reduction_type,
                        }
                    }
                    if reduction_type == "triplet":
                        self.assertRaises(
                            NotImplementedError,
                            lambda: reducer(loss_dict, embeddings, labels),
                        )
                        continue

                    output = reducer(loss_dict, embeddings, labels)
                    if reduction_type == "element":
                        loss_dict = {
                            "loss": {
                                "losses": losses,
                                "indices": c_f.torch_arange_from_size(embeddings),
                                "reduction_type": "element",
                            }
                        }
                    else:
                        anchors = indices[0]
                        correct_output = torch.zeros(
                            batch_size, device=TEST_DEVICE, dtype=dtype
                        )
                        for i in range(len(embeddings)):
                            matching_pairs_mask = anchors == i
                            num_matching_pairs = torch.sum(matching_pairs_mask)
                            if num_matching_pairs > 0:
                                correct_output[i] = (
                                    torch.sum(losses[matching_pairs_mask])
                                    / num_matching_pairs
                                )
                        loss_dict = {
                            "loss": {
                                "losses": correct_output,
                                "indices": c_f.torch_arange_from_size(embeddings),
                                "reduction_type": "element",
                            }
                        }
                    correct_output = inner_reducer(loss_dict, embeddings, labels)
                    self.assertTrue(torch.isclose(output, correct_output, rtol=1e-5))
