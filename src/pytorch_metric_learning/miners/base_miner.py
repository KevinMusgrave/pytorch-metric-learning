from collections.abc import Sequence

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple
from ..utils import _common


class BaseMiner(torch.nn.Module):
    def __init__(self, *, distance: BaseDistance | None = None) -> None:
        super().__init__()
        self.distance = distance if distance is not None else self.default_distance()

    def default_distance(self) -> BaseDistance:
        return LpDistance()

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> IndicesTuple:
        raise NotImplementedError

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: Sequence[Tensor] | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
    ) -> IndicesTuple:
        _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels)
        labels = _common.labels_required(labels)
        candidates = _common.as_indices_tuple(indices_tuple, embeddings.device)
        with torch.no_grad():
            output = self.mine(embeddings, labels, candidates, ref_emb, ref_labels)
        mined = _common.as_indices_tuple(output, embeddings.device)
        if mined is None:
            raise TypeError("Mining must return a tuple of index tensors")
        return mined
