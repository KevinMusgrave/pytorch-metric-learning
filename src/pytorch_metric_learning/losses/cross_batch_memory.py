from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor

from ..miners import BaseMiner
from ..types import IndicesTuple, PairIndices
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction

QUEUE_STATE_KEY = "queue_state"


class _MemoryState(TypedDict):
    queue_idx: int
    has_been_filled: bool


def save_queue_state(module, state_dict, prefix, _local_metadata) -> None:
    state: _MemoryState = {
        "queue_idx": module.queue_idx,
        "has_been_filled": module.has_been_filled,
    }
    state_dict[prefix + QUEUE_STATE_KEY] = state


def load_queue_state(
    module,
    state_dict,
    prefix,
    _local_metadata,
    _strict,
    missing_keys,
    _unexpected_keys,
    _error_msgs,
) -> None:
    state = state_dict.pop(prefix + QUEUE_STATE_KEY, None)
    if state is None:
        missing_keys.append(prefix + QUEUE_STATE_KEY)
        return
    module.queue_idx = state["queue_idx"]
    module.has_been_filled = state["has_been_filled"]


class CrossBatchMemory(torch.nn.Module):
    embedding_memory: Tensor
    label_memory: Tensor

    def __init__(
        self,
        loss: BaseMetricLossFunction,
        embedding_size: int,
        memory_size: int = 1024,
        miner: BaseMiner | None = None,
        *,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        super().__init__()
        if not isinstance(loss, BaseMetricLossFunction):
            raise TypeError("CrossBatchMemory requires a BaseMetricLossFunction")
        self.loss = loss
        self.miner = miner
        self.memory_size = memory_size
        self.register_buffer(
            "embedding_memory", torch.zeros(memory_size, embedding_size)
        )
        self.register_buffer(
            "label_memory", torch.zeros(memory_size, dtype=label_dtype)
        )
        self.reset_queue()
        self.register_state_dict_post_hook(save_queue_state)
        self.register_load_state_dict_pre_hook(load_queue_state)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: Sequence[Tensor] | None = None,
        enqueue_mask: Tensor | None = None,
    ) -> Tensor:
        _common.check_shapes(embeddings, labels)
        _common.check_device(labels, embeddings, "labels")
        if embeddings.shape[1] != self.embedding_memory.shape[1]:
            raise ValueError("embeddings must have the memory's embedding size")
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")
        indices = _common.as_indices_tuple(indices_tuple, embeddings.device)
        if enqueue_mask is not None:
            if enqueue_mask.dtype != torch.bool:
                raise TypeError("enqueue_mask must be a boolean tensor")
            if enqueue_mask.ndim != 1 or len(enqueue_mask) != len(embeddings):
                raise ValueError("enqueue_mask must contain one value per embedding")
            _common.check_device(enqueue_mask, embeddings, "enqueue_mask")
        _common.check_parameter_placement(self, self.embedding_memory, embeddings)
        embeddings = embeddings.to(dtype=self.embedding_memory.dtype)
        labels = self.stored_labels(labels)

        if enqueue_mask is not None:
            emb_for_queue = embeddings[enqueue_mask]
            labels_for_queue = labels[enqueue_mask]
            embeddings = embeddings[~enqueue_mask]
            labels = labels[~enqueue_mask]
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
        count = len(emb_for_queue)
        if count > self.memory_size:
            raise ValueError(
                "The number of enqueued embeddings must not exceed memory_size"
            )

        slots = (
            torch.arange(self.queue_idx, self.queue_idx + count, device=labels.device)
            % self.memory_size
        )
        filled = self.has_been_filled or self.queue_idx + count >= self.memory_size
        memory_end = self.memory_size if filled else self.queue_idx + count
        E_mem = self.embedding_memory[:memory_end].clone()
        L_mem = self.label_memory[:memory_end].clone()
        E_mem[slots] = emb_for_queue.detach()
        L_mem[slots] = labels_for_queue
        mined = self.create_indices_tuple(
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices,
            None if enqueue_mask is not None else slots,
        )
        loss = self.loss(embeddings, labels, mined, E_mem, L_mem)
        self.embedding_memory[slots] = emb_for_queue.detach()
        self.label_memory[slots] = labels_for_queue
        self.queue_idx = (self.queue_idx + count) % self.memory_size
        self.has_been_filled = filled
        return loss

    def stored_labels(self, labels: Tensor) -> Tensor:
        dtype = self.label_memory.dtype
        if labels.dtype == dtype:
            return labels
        converted = labels.to(dtype)
        if not torch.equal(converted.to(labels.dtype), labels):
            raise ValueError(
                f"CrossBatchMemory stores {dtype} labels and cannot represent these "
                f"{labels.dtype} labels exactly; set label_dtype"
            )
        return converted

    def create_indices_tuple(
        self,
        embeddings: Tensor,
        labels: Tensor,
        E_mem: Tensor,
        L_mem: Tensor,
        input_indices_tuple: IndicesTuple | None,
        curr_batch_idx: Tensor | None,
    ) -> IndicesTuple:
        if self.miner:
            indices_tuple = self.miner(
                embeddings, labels, ref_emb=E_mem, ref_labels=L_mem
            )
        else:
            indices_tuple = indexing.get_all_pairs_indices(labels, L_mem)
        if curr_batch_idx is not None:
            indices_tuple = indexing.remove_self_comparisons(
                indices_tuple, curr_batch_idx
            )
        if input_indices_tuple is None:
            return indices_tuple
        if isinstance(indices_tuple, PairIndices):
            input_indices_tuple = indexing.convert_to_pairs(input_indices_tuple, labels)
        else:
            input_indices_tuple = indexing.convert_to_triplets(
                input_indices_tuple, labels
            )
        return indexing.concatenate_indices_tuples(indices_tuple, input_indices_tuple)

    def reset_queue(self) -> None:
        self.embedding_memory.zero_()
        self.label_memory.zero_()
        self.has_been_filled = False
        self.queue_idx = 0
