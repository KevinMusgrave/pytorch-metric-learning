import inspect

import pytest
import torch

from pytorch_metric_learning import miners
from pytorch_metric_learning.types import PairIndices, TripletIndices

from .. import TEST_DEVICE

MINERS = [
    getattr(miners, name)
    for name in miners.__all__
    if not inspect.isabstract(getattr(miners, name)) and name != "BaseMiner"
]


@pytest.mark.parametrize("miner_class", MINERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("query_size,reference_size", [(0, None), (3, 0), (0, 3)])
def test_empty_query_or_reference_mines_nothing(
    miner_class, query_size, reference_size
):
    miner = miner_class()
    embeddings = torch.randn(query_size, 4, device=TEST_DEVICE)
    labels = torch.arange(query_size, device=TEST_DEVICE) % 2
    reference = None
    reference_labels = None
    if reference_size is not None:
        reference = torch.randn(reference_size, 4, device=TEST_DEVICE)
        reference_labels = torch.arange(reference_size, device=TEST_DEVICE) % 2
    mined = miner(embeddings, labels, ref_emb=reference, ref_labels=reference_labels)
    assert isinstance(mined, (PairIndices, TripletIndices))
    assert all(len(indices) == 0 for indices in mined)
    assert all(indices.device.type == TEST_DEVICE.type for indices in mined)
