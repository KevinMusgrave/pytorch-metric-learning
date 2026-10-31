from collections.abc import Callable, Sequence
from typing import ClassVar, cast

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple
from ..utils import _common, indexing


class BaseMiner(torch.nn.Module):
    supports_mat: ClassVar[bool] = False

    def __init__(
        self,
        *,
        distance: BaseDistance | None = None,
        tuples_per_anchor: int | None = None,
    ) -> None:
        super().__init__()
        if tuples_per_anchor is not None and tuples_per_anchor <= 0:
            raise ValueError("tuples_per_anchor must be positive")
        self.distance = distance if distance is not None else self.default_distance()
        self.tuples_per_anchor = tuples_per_anchor

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
        *,
        mat: Tensor | None = None,
    ) -> IndicesTuple:
        _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels)
        labels = _common.labels_required(labels)
        candidates = _common.as_indices_tuple(indices_tuple, embeddings.device)
        with torch.no_grad():
            if mat is None:
                output = self.mine(embeddings, labels, candidates, ref_emb, ref_labels)
            else:
                _common.check_matrix(self, mat, embeddings, ref_emb)
                mine = cast(Callable[..., IndicesTuple], self.mine)
                output = mine(
                    embeddings, labels, candidates, ref_emb, ref_labels, mat=mat
                )
        mined = _common.as_indices_tuple(output, embeddings.device)
        if mined is None:
            raise TypeError("Mining must return a tuple of index tensors")
        if self.tuples_per_anchor is None:
            return mined
        return indexing._limit_per_anchor(mined, self.tuples_per_anchor)
