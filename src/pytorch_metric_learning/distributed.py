from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
from torch import Tensor
from torch.distributed import _functional_collectives

from .losses import BaseMetricLossFunction, CrossBatchMemory, MultipleLosses
from .miners import BaseMiner
from .types import DistributedObjective, IndicesTuple, MetricLoss, PairIndices
from .utils import _common, indexing

__all__ = ["DistributedLossWrapper", "DistributedMinerWrapper"]

_OBJECTIVES: tuple[DistributedObjective, ...] = ("global", "local_anchor")

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


_SCHEMA_WIDTH = 6
_INVALID_SCHEMA = [-1] * _SCHEMA_WIDTH


class _Gathered(NamedTuple):
    embeddings: Tensor
    labels: Tensor | None
    offset: int


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _check_objective(objective: str) -> None:
    if objective not in _OBJECTIVES:
        raise ValueError('objective must be "global" or "local_anchor"')


def _dtype_index(tensor: Tensor) -> int:
    return _DTYPES.index(tensor.dtype) if tensor.dtype in _DTYPES else -1


def _gather_metadata(metadata: Tensor) -> list[list[int]]:
    rows = [
        torch.empty_like(metadata) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(rows, metadata)
    return torch.stack(rows).cpu().tolist()


def _local_error(check: Callable[[], None]) -> Exception | None:
    try:
        check()
    except (TypeError, ValueError) as error:
        return error
    return None


def _check_embeddings(emb: Tensor, name: str) -> None:
    if emb.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D tensor of shape (batch_size, embedding_size)"
        )


def _schema(emb: Tensor | None, labels: Tensor | None) -> list[int]:
    if emb is None:
        return [0] * _SCHEMA_WIDTH
    return [
        emb.shape[0],
        emb.shape[1],
        _dtype_index(emb),
        _dtype_index(labels) if labels is not None else -1,
        int(labels is not None),
        int(emb.requires_grad),
    ]


def _synchronize(
    device: torch.device,
    flags: list[bool],
    error: Exception | None,
    inputs: Sequence[tuple[Tensor | None, Tensor | None]],
) -> list[list[int]]:
    values = [
        value
        for emb, labels in inputs
        for value in (_schema(emb, labels) if error is None else _INVALID_SCHEMA)
    ]
    metadata = torch.tensor(
        [*flags, error is None, *values], dtype=torch.long, device=device
    )
    rows = _gather_metadata(metadata)
    count = len(flags)
    if any(row[:count] != rows[0][:count] for row in rows):
        raise ValueError(
            "Ranks must use the same wrapper objective and optional arguments"
        )
    invalid = [rank for rank, row in enumerate(rows) if not row[count]]
    if error is not None:
        raise error
    if invalid:
        raise ValueError(f"Inputs on ranks {invalid} are invalid")
    sizes = []
    for index, (emb, _) in enumerate(inputs):
        start = count + 1 + index * _SCHEMA_WIDTH
        schemas = [row[start : start + _SCHEMA_WIDTH] for row in rows]
        if emb is not None and any(
            schema[2] not in range(4)
            or (schema[4] and schema[3] < 0)
            or schema[1:] != schemas[0][1:]
            for schema in schemas
        ):
            raise ValueError(
                "Ranks must provide embeddings with matching dimensions, dtypes, "
                "and requires_grad, and labels with matching dtypes"
            )
        sizes.append([schema[0] for schema in schemas])
    return sizes


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


def _gather(
    emb: Tensor, labels: Tensor | None, sizes: list[int], remote_gradients: bool = False
) -> _Gathered:
    all_emb = _gather_tensor(emb, sizes, remote_gradients)
    all_labels = _gather_tensor(labels, sizes) if labels is not None else None
    offset = sum(sizes[: torch.distributed.get_rank()])
    return _Gathered(all_emb, all_labels, offset)


def _local_anchor_pairs(
    labels: Tensor, ref_labels: Tensor, offset: int, remove_self: bool
) -> PairIndices:
    pairs = indexing.get_all_pairs_indices(labels, ref_labels)
    if not remove_self:
        return pairs
    curr_batch_idx = torch.arange(len(labels), device=labels.device) + offset
    return indexing._remove_self_comparisons(pairs, curr_batch_idx)


class DistributedLossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss: MetricLoss | CrossBatchMemory,
        objective: DistributedObjective = "global",
    ) -> None:
        super().__init__()
        if not isinstance(
            loss, (BaseMetricLossFunction, MultipleLosses, CrossBatchMemory)
        ):
            raise TypeError(
                "The input loss must be a BaseMetricLossFunction, MultipleLosses, "
                "or CrossBatchMemory"
            )
        _check_objective(objective)
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
        size = _world_size()
        memory = isinstance(self.loss, CrossBatchMemory)
        error = _local_error(
            lambda: self.validate(
                embeddings, labels, indices_tuple, ref_emb, ref_labels, enqueue_mask
            )
        )
        if size <= 1:
            if error is not None:
                raise error
            if isinstance(self.loss, CrossBatchMemory):
                return self.loss(embeddings, labels, indices_tuple, enqueue_mask)
            return self.loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        flags = [
            self.objective == "local_anchor",
            memory,
            labels is not None,
            indices_tuple is not None,
            ref_emb is not None,
            ref_labels is not None,
            enqueue_mask is not None,
        ]
        sizes, ref_sizes = _synchronize(
            embeddings.device,
            flags,
            error,
            ((embeddings, labels), (ref_emb, ref_labels)),
        )
        if isinstance(self.loss, CrossBatchMemory):
            return self.memory_loss(
                self.loss, embeddings, labels, indices_tuple, enqueue_mask, sizes
            )
        return self.regular_loss(
            self.loss,
            embeddings,
            labels,
            indices_tuple,
            ref_emb,
            ref_labels,
            sizes,
            ref_sizes,
        )

    def validate(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: Sequence[Tensor] | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        enqueue_mask: Tensor | None,
    ) -> None:
        if not isinstance(self.loss, CrossBatchMemory):
            if enqueue_mask is not None:
                raise ValueError("enqueue_mask requires a CrossBatchMemory loss")
        elif ref_emb is not None or ref_labels is not None:
            raise ValueError(
                "CrossBatchMemory is not compatible with ref_emb and ref_labels"
            )
        elif labels is None:
            raise ValueError("CrossBatchMemory requires labels")
        _check_embeddings(embeddings, "embeddings")
        if ref_emb is not None:
            _check_embeddings(ref_emb, "ref_emb")
        _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels)
        _common.as_indices_tuple(indices_tuple, embeddings.device)
        if enqueue_mask is not None and not (
            enqueue_mask.dtype == torch.bool
            and enqueue_mask.ndim == 1
            and len(enqueue_mask) == len(embeddings)
            and enqueue_mask.device == embeddings.device
        ):
            raise ValueError(
                "enqueue_mask must be a boolean vector with one entry per embedding "
                "on the embeddings' device"
            )

    def regular_loss(
        self,
        loss: MetricLoss,
        emb: Tensor,
        labels: Tensor | None,
        indices_tuple: Sequence[Tensor] | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        sizes: list[int],
        ref_sizes: list[int],
    ) -> Tensor:
        local_anchor = self.objective == "local_anchor"
        query = _gather(emb, labels, sizes, remote_gradients=not local_anchor)
        reference = None
        if ref_emb is not None:
            reference = _gather(
                ref_emb, ref_labels, ref_sizes, remote_gradients=not local_anchor
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
            indices_tuple = _local_anchor_pairs(
                labels, all_ref.labels, query.offset, remove_self=reference is None
            )
        local = loss(emb, labels, indices_tuple, all_ref.embeddings, all_ref.labels)
        return local * len(sizes)

    def memory_loss(
        self,
        loss: CrossBatchMemory,
        emb: Tensor,
        labels: Tensor | None,
        indices_tuple: Sequence[Tensor] | None,
        enqueue_mask: Tensor | None,
        sizes: list[int],
    ) -> Tensor:
        local_anchor = self.objective == "local_anchor"
        query = _gather(emb, labels, sizes, remote_gradients=not local_anchor)
        if enqueue_mask is not None:
            enqueue_mask = _gather_tensor(enqueue_mask, sizes)
        if query.labels is None or labels is None:
            raise ValueError("CrossBatchMemory requires labels")
        if not local_anchor:
            return loss(query.embeddings, query.labels, indices_tuple, enqueue_mask)
        local = loss.forward_local_anchors(
            emb,
            labels,
            query.embeddings,
            query.labels,
            query.offset,
            indices_tuple,
            enqueue_mask,
        )
        return local * len(sizes)


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(
        self, miner: BaseMiner, objective: DistributedObjective = "global"
    ) -> None:
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError("The input miner must extend BaseMiner")
        _check_objective(objective)
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
        size = _world_size()
        if size <= 1:
            return self.miner(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        flags = [
            self.objective == "local_anchor",
            indices_tuple is not None,
            ref_emb is not None,
            ref_labels is not None,
        ]
        error = _local_error(
            lambda: self.validate(
                embeddings, labels, indices_tuple, ref_emb, ref_labels
            )
        )
        sizes, ref_sizes = _synchronize(
            embeddings.device,
            flags,
            error,
            ((embeddings, labels), (ref_emb, ref_labels)),
        )
        query = _gather(embeddings, labels, sizes)
        reference = None
        if ref_emb is not None:
            reference = _gather(ref_emb, ref_labels, ref_sizes)
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
            candidates = _local_anchor_pairs(
                labels, all_ref.labels, query.offset, remove_self=True
            )
        return self.miner(
            embeddings, labels, candidates, all_ref.embeddings, all_ref.labels
        )

    def validate(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: Sequence[Tensor] | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> None:
        _check_embeddings(embeddings, "embeddings")
        if ref_emb is not None:
            _check_embeddings(ref_emb, "ref_emb")
        _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels)
        _common.labels_required(labels)
        if indices_tuple is not None:
            raise ValueError("Candidate indices_tuple is not supported across ranks")
