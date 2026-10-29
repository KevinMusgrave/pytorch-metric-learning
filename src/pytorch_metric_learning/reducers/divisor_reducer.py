import torch
from torch import Tensor

from ..types import LossTerm
from ..utils._common import accumulation_dtype
from .base_reducer import BaseReducer, validate_term


class DivisorReducer(BaseReducer):
    def reduce(
        self, term: LossTerm, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        losses, anchors = validate_term(term)
        if anchors is None:
            return losses
        return self.sum_and_divide(losses, term.get("divisor", 1))

    def sum_and_divide(self, losses: Tensor, divisor: Tensor | float) -> Tensor:
        dtype = accumulation_dtype(losses.dtype)
        divisor = torch.as_tensor(divisor, device=losses.device).to(dtype)
        total = losses.sum(dtype=dtype)
        safe = torch.where(divisor != 0, divisor, torch.ones_like(divisor))
        output = torch.where(divisor != 0, total / safe, torch.zeros_like(total))
        return output.to(losses.dtype)
