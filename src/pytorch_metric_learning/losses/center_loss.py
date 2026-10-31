import torch
from torch import Tensor

from ..reducers import BaseReducer
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class CenterLoss(BaseMetricLossFunction):
    centers: Tensor

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        alpha: float = 0.5,
        *,
        reducer: BaseReducer | None = None,
        embedding_regularizer: BaseRegularizer | None = None,
        embedding_reg_weight: float = 1.0,
    ) -> None:
        super().__init__(
            reducer=reducer,
            embedding_regularizer=embedding_regularizer,
            embedding_reg_weight=embedding_reg_weight,
        )
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.register_buffer("centers", torch.randn(num_classes, embedding_size))

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.indices_tuple_not_supported(indices_tuple)
        _common.ref_not_supported(ref_emb)
        offsets = self.offsets_from_centers(embeddings, labels)
        return {
            "loss": {
                "losses": offsets.pow(2).sum(dim=1) / 2,
                "indices": _common.arange_like(embeddings),
            }
        }

    @torch.no_grad()
    def update_centers(self, embeddings: Tensor, labels: Tensor) -> None:
        _common.check_shapes(embeddings, labels)
        offsets = self.offsets_from_centers(embeddings, labels)
        classes, class_index, counts = torch.unique(
            labels, return_inverse=True, return_counts=True
        )
        totals = offsets.new_zeros(len(classes), offsets.shape[1])
        totals.index_add_(0, class_index, offsets)
        step = self.alpha * totals / (1 + counts).unsqueeze(1)
        self.centers[classes] += step.to(self.centers.dtype)

    def offsets_from_centers(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        _common.check_parameter_placement(self, self.centers, embeddings)
        if embeddings.shape[1] != self.embedding_size:
            raise ValueError("Embedding dimension must match embedding_size")
        if len(labels) and not 0 <= labels.min() <= labels.max() < self.num_classes:
            raise ValueError("labels must be in [0, num_classes)")
        dtype = _common.accumulation_dtype(embeddings.dtype)
        return embeddings.to(dtype) - self.centers[labels].to(dtype)
