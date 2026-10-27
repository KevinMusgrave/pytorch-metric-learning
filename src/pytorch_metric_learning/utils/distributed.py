import warnings

import torch
from torch.distributed import _functional_collectives

from ..losses import BaseMetricLossFunction, CrossBatchMemory
from ..miners import BaseMiner
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu

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


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dtype_index(tensor: torch.Tensor) -> int:
    return _DTYPES.index(tensor.dtype) if tensor.dtype in _DTYPES else -1


def _gather_metadata(metadata: torch.Tensor) -> list[list[int]]:
    rows = [
        torch.empty_like(metadata) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(rows, metadata)
    return torch.stack(rows).cpu().tolist()


def _batch_sizes(emb: torch.Tensor, labels: torch.Tensor | None) -> list[int]:
    label_shape_valid = labels is None or (
        emb.ndim > 0 and labels.ndim == 1 and len(labels) == len(emb)
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
                "and requires_grad, and consistent labels with one label per embedding"
            )
    return [row[0] for row in rows]


def _gather_tensor(
    tensor: torch.Tensor, sizes: list[int], remote_gradients: bool = False
) -> torch.Tensor:
    padded_size = max(max(sizes), 1)
    padding = tensor.new_zeros((padded_size - len(tensor), *tensor.shape[1:]))
    padded = torch.cat((tensor, padding)).contiguous()
    if remote_gradients:
        group = torch.distributed.group.WORLD
        assert group is not None
        combined = _functional_collectives.all_gather_single(padded, 0, group)
        rows = combined.split(padded_size)
    else:
        rows = [torch.empty_like(padded) for _ in sizes]
        torch.distributed.all_gather(rows, padded)
        rows[torch.distributed.get_rank()] = padded
    return torch.cat([row[:size] for row, size in zip(rows, sizes, strict=True)])


def _require_same_arguments(emb: torch.Tensor, flags: list[bool]) -> None:
    rows = _gather_metadata(torch.tensor(flags, device=emb.device, dtype=torch.long))
    if any(row != rows[0] for row in rows):
        raise ValueError("Ranks must use the same wrapper mode and optional arguments")


def gather(emb, labels, remote_gradients=False):
    sizes = _batch_sizes(emb, labels)
    if labels is not None:
        labels = labels.to(emb.device)
    all_emb = _gather_tensor(emb, sizes, remote_gradients)
    all_labels = _gather_tensor(labels, sizes) if labels is not None else None
    offset = sum(sizes[: torch.distributed.get_rank()])
    return all_emb, all_labels, labels, offset, sizes


def gather_emb_and_ref(
    emb, labels, ref_emb=None, ref_labels=None, remote_gradients=False
):
    all_emb, all_labels, labels, offset, sizes = gather(emb, labels, remote_gradients)
    all_ref_emb, all_ref_labels = None, None

    if ref_emb is not None:
        all_ref_emb, all_ref_labels, _, _, _ = gather(
            ref_emb, ref_labels, remote_gradients
        )

    return all_emb, all_labels, all_ref_emb, all_ref_labels, labels, offset, sizes


def get_indices_tuple(
    labels,
    ref_labels,
    embeddings=None,
    ref_emb=None,
    miner=None,
    *,
    offset=0,
    remove_self=True,
):
    device = labels.device
    curr_batch_idx = torch.arange(len(labels), device=device) + offset
    if miner:
        indices_tuple = miner(embeddings, labels, ref_emb, ref_labels)
    else:
        indices_tuple = lmu.get_all_pairs_indices(labels, ref_labels)
    if remove_self and len(labels):
        return lmu.remove_self_comparisons(
            indices_tuple, curr_batch_idx, len(ref_labels)
        )
    return indices_tuple


def gather_enqueue_mask(enqueue_mask, device, sizes):
    if enqueue_mask is None:
        return enqueue_mask
    enqueue_mask = c_f.to_device(enqueue_mask, device=device)
    return _gather_tensor(enqueue_mask, sizes)


def select_ref_or_regular(regular, ref):
    return regular if ref is None else ref


class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss, efficient=False):
        super().__init__()
        if not isinstance(loss, (BaseMetricLossFunction, CrossBatchMemory)):
            raise TypeError(
                "The input loss must extend BaseMetricLossFunction or CrossBatchMemory"
            )
        if isinstance(loss, CrossBatchMemory) and efficient:
            raise ValueError(
                "CrossBatchMemory with efficient=True is not currently supported"
            )
        self.loss = loss
        self.efficient = efficient

    def forward(
        self,
        embeddings,
        labels=None,
        indices_tuple=None,
        ref_emb=None,
        ref_labels=None,
        enqueue_mask=None,
    ):
        if is_distributed():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
            warnings.warn(
                "DistributedLossWrapper is being used in a non-distributed setting. "
                "Returning the loss as is.",
                stacklevel=2,
            )

        valid_mask = enqueue_mask is None or (
            enqueue_mask.dtype == torch.bool
            and enqueue_mask.ndim == 1
            and embeddings.ndim > 0
            and len(enqueue_mask) == len(embeddings)
        )
        if world_size > 1:
            _require_same_arguments(
                embeddings,
                [
                    self.efficient,
                    isinstance(self.loss, CrossBatchMemory),
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
                "enqueue_mask must be a boolean vector with one entry per embedding"
            )

        common_args = (
            embeddings,
            labels,
            indices_tuple,
            ref_emb,
            ref_labels,
            world_size,
        )
        if isinstance(self.loss, CrossBatchMemory):
            return self.forward_cross_batch(*common_args, enqueue_mask)
        if enqueue_mask is not None:
            raise ValueError("enqueue_mask requires a CrossBatchMemory loss")
        return self.forward_regular_loss(*common_args)

    def forward_regular_loss(
        self, emb, labels, indices_tuple, ref_emb, ref_labels, world_size
    ):
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, ref_emb, ref_labels)

        all_emb, all_labels, all_ref_emb, all_ref_labels, labels, offset, _ = (
            gather_emb_and_ref(emb, labels, ref_emb, ref_labels, not self.efficient)
        )

        if self.efficient:
            if all_labels is not None:
                all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            if indices_tuple is None:
                indices_tuple = get_indices_tuple(
                    labels, all_labels, offset=offset, remove_self=ref_emb is None
                )
            loss = self.loss(emb, labels, indices_tuple, all_emb, all_labels)
        else:
            loss = self.loss(
                all_emb, all_labels, indices_tuple, all_ref_emb, all_ref_labels
            )

        return loss * world_size if self.efficient else loss

    def forward_cross_batch(
        self,
        emb,
        labels,
        indices_tuple,
        ref_emb,
        ref_labels,
        world_size,
        enqueue_mask,
    ):
        if ref_emb is not None or ref_labels is not None:
            raise ValueError(
                "CrossBatchMemory is not compatible with ref_emb and ref_labels"
            )
        if labels is None:
            raise ValueError("CrossBatchMemory requires labels")

        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, enqueue_mask)

        all_emb, all_labels, _, _, _, _, sizes = gather_emb_and_ref(
            emb, labels, remote_gradients=True
        )
        enqueue_mask = gather_enqueue_mask(enqueue_mask, emb.device, sizes)
        loss = self.loss(all_emb, all_labels, indices_tuple, enqueue_mask)
        return loss


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner, efficient=False):
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError("The input miner must extend BaseMiner")
        self.miner = miner
        self.efficient = efficient

    def forward(self, emb, labels, ref_emb=None, ref_labels=None):
        world_size = torch.distributed.get_world_size() if is_distributed() else 1
        if world_size <= 1:
            return self.miner(emb, labels, ref_emb, ref_labels)

        _require_same_arguments(
            emb, [self.efficient, ref_emb is not None, ref_labels is not None]
        )
        all_emb, all_labels, all_ref_emb, all_ref_labels, labels, offset, _ = (
            gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        )
        if self.efficient:
            all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            return get_indices_tuple(
                labels,
                all_labels,
                emb,
                all_emb,
                self.miner,
                offset=offset,
                remove_self=ref_emb is None,
            )
        else:
            return self.miner(all_emb, all_labels, all_ref_emb, all_ref_labels)
