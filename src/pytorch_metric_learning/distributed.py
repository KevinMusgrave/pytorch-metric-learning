from collections.abc import Sequence
from typing import Literal, NamedTuple

import torch
from torch import Tensor
from torch.distributed import _functional_collectives

from .losses import BaseMetricLossFunction, CrossBatchMemory
from .miners import BaseMiner
from .types import IndicesTuple, PairIndices
from .utils import _common, indexing

__all__ = ["DistributedLossWrapper", "DistributedMinerWrapper"]

type Objective = Literal["global", "local_anchor"]
OBJECTIVES = ("global", "local_anchor")

_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)


class Gathered(NamedTuple):
    embeddings: Tensor
    labels: Tensor | None
    offset: int


def world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def check_objective(objective: str) -> None:
    if objective not in OBJECTIVES:
        raise ValueError('objective must be "global" or "local_anchor"')


def _dtype_index(tensor: Tensor) -> int:
    return _DTYPES.index(tensor.dtype) if tensor.dtype in _DTYPES else -1


def _gather_metadata(metadata: Tensor) -> list[list[int]]:
    rows = [
        torch.empty_like(metadata) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(rows, metadata)
    return torch.stack(rows).cpu().tolist()


def _batch_sizes(emb: Tensor, labels: Tensor | None) -> list[int]:
    label_shape_valid = labels is None or (
        emb.ndim > 0
        and labels.ndim == 1
        and len(labels) == len(emb)
        and labels.device == emb.device
    )
    metadata = torch.tensor(
        [
            emb.shape[0] if emb.ndim else 0,
            emb.shape[1] if emb.ndim == 2 else -1,
            _dtype_index(emb),
            _dtype_index(labels) if labels is not None else -1,
            int(labels is not None),
            int(emb.requires_grad),
            int(label_shape_valid),
        ],
        device=emb.device,
    )
    rows = _gather_metadata(metadata)
    for row in rows:
        if (
            row[1] < 0
            or row[2] not in range(4)
            or row[6] == 0
            or (row[4] and row[3] < 0)
            or row[1:6] != rows[0][1:6]
        ):
            raise ValueError(
                "Ranks must provide 2D embeddings with matching dimensions, dtypes, "
                "and requires_grad, and consistent labels with one label per "
                "embedding on the embeddings' device"
            )
    return [row[0] for row in rows]


def _gather_tensor(
    tensor: Tensor, sizes: list[int], remote_gradients: bool = False
) -> Tensor:
    padded_size = max(max(sizes), 1)
    padding = tensor.new_zeros((padded_size - len(tensor), *tensor.shape[1:]))
    padded = torch.cat((tensor, padding)).contiguous()
    if remote_gradients:
        group = torch.distributed.group.WORLD
        if group is None:
            raise RuntimeError("A distributed process group must be initialized")
        combined = _functional_collectives.all_gather_single(padded, 0, group)
        rows = combined.split(padded_size)
    else:
        rows = [torch.empty_like(padded) for _ in sizes]
        torch.distributed.all_gather(rows, padded)
        rows[torch.distributed.get_rank()] = padded
    return torch.cat([row[:size] for row, size in zip(rows, sizes, strict=True)])


def _require_same_arguments(emb: Tensor, flags: list[bool]) -> None:
    rows = _gather_metadata(torch.tensor(flags, device=emb.device, dtype=torch.long))
    if any(row != rows[0] for row in rows):
        raise ValueError(
            "Ranks must use the same wrapper objective and optional arguments"
        )


def gather(
    emb: Tensor, labels: Tensor | None, remote_gradients: bool = False
) -> tuple[Gathered, list[int]]:
    sizes = _batch_sizes(emb, labels)
    all_emb = _gather_tensor(emb, sizes, remote_gradients)
    all_labels = _gather_tensor(labels, sizes) if labels is not None else None
    offset = sum(sizes[: torch.distributed.get_rank()])
    return Gathered(all_emb, all_labels, offset), sizes


def local_anchor_pairs(
    labels: Tensor, ref_labels: Tensor, offset: int, remove_self: bool
) -> PairIndices:
    pairs = indexing.get_all_pairs_indices(labels, ref_labels)
    if not remove_self:
        return pairs
    curr_batch_idx = torch.arange(len(labels), device=labels.device) + offset
    return indexing.remove_self_comparisons(pairs, curr_batch_idx)


class DistributedLossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss: BaseMetricLossFunction | CrossBatchMemory,
        objective: Objective = "global",
    ) -> None:
        super().__init__()
        if not isinstance(loss, (BaseMetricLossFunction, CrossBatchMemory)):
            raise TypeError(
                "The input loss must extend BaseMetricLossFunction or CrossBatchMemory"
            )
        check_objective(objective)
        if isinstance(loss, CrossBatchMemory) and objective == "local_anchor":
            raise ValueError("CrossBatchMemory supports only the global objective")
        self.loss = loss
        self.objective = objective

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        indices_tuple: Sequence[Tensor] | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
        enqueue_mask: Tensor | None = None,
    ) -> Tensor:
        size = world_size()
        memory = isinstance(self.loss, CrossBatchMemory)
        valid_mask = enqueue_mask is None or (
            enqueue_mask.dtype == torch.bool
            and enqueue_mask.ndim == 1
            and embeddings.ndim > 0
            and len(enqueue_mask) == len(embeddings)
            and enqueue_mask.device == embeddings.device
        )
        if size > 1:
            _require_same_arguments(
                embeddings,
                [
                    self.objective == "local_anchor",
                    memory,
                    labels is not None,
                    indices_tuple is not None,
                    ref_emb is not None,
                    ref_labels is not None,
                    enqueue_mask is not None,
                    valid_mask,
                ],
            )
        if not valid_mask:
            raise ValueError(
                "enqueue_mask must be a boolean vector with one entry per embedding "
                "on the embeddings' device"
            )
        if isinstance(self.loss, CrossBatchMemory):
            if ref_emb is not None or ref_labels is not None:
                raise ValueError(
                    "CrossBatchMemory is not compatible with ref_emb and ref_labels"
                )
            if labels is None:
                raise ValueError("CrossBatchMemory requires labels")
            return self.memory_loss(
                self.loss, embeddings, labels, indices_tuple, enqueue_mask, size
            )
        if enqueue_mask is not None:
            raise ValueError("enqueue_mask requires a CrossBatchMemory loss")
        return self.regular_loss(
            self.loss, embeddings, labels, indices_tuple, ref_emb, ref_labels, size
        )

    def regular_loss(
        self,
        loss: BaseMetricLossFunction,
        emb: Tensor,
        labels: Tensor | None,
        indices_tuple: Sequence[Tensor] | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        size: int,
    ) -> Tensor:
        if size <= 1:
            return loss(emb, labels, indices_tuple, ref_emb, ref_labels)
        _common.check_device(labels, emb, "labels")
        _common.check_device(ref_labels, emb, "ref_labels")
        local_anchor = self.objective == "local_anchor"
        query, _ = gather(emb, labels, remote_gradients=not local_anchor)
        reference = None
        if ref_emb is not None:
            reference, _ = gather(
                ref_emb, ref_labels, remote_gradients=not local_anchor
            )
        if not local_anchor:
            return loss(
                query.embeddings,
                query.labels,
                indices_tuple,
                None if reference is None else reference.embeddings,
                None if reference is None else reference.labels,
            )
        all_ref = query if reference is None else reference
        if indices_tuple is None and labels is not None and all_ref.labels is not None:
            indices_tuple = local_anchor_pairs(
                labels, all_ref.labels, query.offset, remove_self=reference is None
            )
        local = loss(emb, labels, indices_tuple, all_ref.embeddings, all_ref.labels)
        return local * size

    def memory_loss(
        self,
        loss: CrossBatchMemory,
        emb: Tensor,
        labels: Tensor,
        indices_tuple: Sequence[Tensor] | None,
        enqueue_mask: Tensor | None,
        size: int,
    ) -> Tensor:
        if size <= 1:
            return loss(emb, labels, indices_tuple, enqueue_mask)
        _common.check_device(labels, emb, "labels")
        query, sizes = gather(emb, labels, remote_gradients=True)
        if enqueue_mask is not None:
            enqueue_mask = _gather_tensor(enqueue_mask, sizes)
        if query.labels is None:
            raise ValueError("CrossBatchMemory requires labels")
        return loss(query.embeddings, query.labels, indices_tuple, enqueue_mask)


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner: BaseMiner, objective: Objective = "global") -> None:
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError("The input miner must extend BaseMiner")
        check_objective(objective)
        self.miner = miner
        self.objective = objective

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: Sequence[Tensor] | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
    ) -> IndicesTuple:
        size = world_size()
        if size <= 1:
            return self.miner(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        _require_same_arguments(
            embeddings,
            [
                self.objective == "local_anchor",
                indices_tuple is not None,
                ref_emb is not None,
                ref_labels is not None,
            ],
        )
        if indices_tuple is not None:
            raise ValueError("Candidate indices_tuple is not supported across ranks")
        _common.check_device(labels, embeddings, "labels")
        _common.check_device(ref_labels, embeddings, "ref_labels")
        query, _ = gather(embeddings, labels)
        reference = None
        if ref_emb is not None:
            reference, _ = gather(ref_emb, ref_labels)
        if self.objective == "global":
            return self.miner(
                query.embeddings,
                query.labels,
                ref_emb=None if reference is None else reference.embeddings,
                ref_labels=None if reference is None else reference.labels,
            )
        all_ref = query if reference is None else reference
        if all_ref.labels is None:
            raise ValueError("Mining requires labels")
        candidates = None
        if reference is None:
            candidates = local_anchor_pairs(
                labels, all_ref.labels, query.offset, remove_self=True
            )
        return self.miner(
            embeddings, labels, candidates, all_ref.embeddings, all_ref.labels
        )
