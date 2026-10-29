from typing import TypedDict

import torch
from torch import Tensor

from ..miners import BaseMiner
from ..types import IndicesTuple
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
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.label_dtype = label_dtype
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
        indices_tuple: IndicesTuple | None = None,
        enqueue_mask: Tensor | None = None,
    ) -> Tensor:
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")
        if enqueue_mask is not None:
            if enqueue_mask.dtype != torch.bool:
                raise TypeError("enqueue_mask must be a boolean tensor")
            if enqueue_mask.ndim != 1:
                raise ValueError("enqueue_mask must be one-dimensional")
            if len(enqueue_mask) != len(embeddings):
                raise ValueError("enqueue_mask must contain one value per embedding")
        _common.check_parameter_placement(self, self.embedding_memory, embeddings)
        embeddings = embeddings.to(dtype=self.embedding_memory.dtype)
        if not torch.can_cast(labels.dtype, self.label_memory.dtype):
            raise ValueError(
                f"CrossBatchMemory stores {self.label_memory.dtype} labels and cannot "
                f"safely store {labels.dtype} labels; set label_dtype"
            )
        labels = labels.to(device=embeddings.device, dtype=self.label_memory.dtype)

        if enqueue_mask is not None:
            emb_for_queue = embeddings[enqueue_mask]
            labels_for_queue = labels[enqueue_mask]
            embeddings = embeddings[~enqueue_mask]
            labels = labels[~enqueue_mask]
            do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True

        self.add_to_memory(emb_for_queue, labels_for_queue, len(emb_for_queue))
        memory_end = self.memory_size if self.has_been_filled else self.queue_idx
        E_mem = self.embedding_memory[:memory_end].clone()
        L_mem = self.label_memory[:memory_end].clone()
        indices_tuple = self.create_indices_tuple(
            embeddings, labels, E_mem, L_mem, indices_tuple, do_remove_self_comparisons
        )
        return self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)

    def add_to_memory(
        self, embeddings: Tensor, labels: Tensor, batch_size: int
    ) -> None:
        if batch_size > self.memory_size:
            raise ValueError(
                "The number of enqueued embeddings must not exceed memory_size"
            )
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        if batch_size == 0:
            return
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def create_indices_tuple(
        self,
        embeddings: Tensor,
        labels: Tensor,
        E_mem: Tensor,
        L_mem: Tensor,
        input_indices_tuple: IndicesTuple | None,
        do_remove_self_comparisons: bool,
    ) -> IndicesTuple:
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = indexing.get_all_pairs_indices(labels, L_mem)
        if do_remove_self_comparisons:
            indices_tuple = indexing.remove_self_comparisons(
                indices_tuple, self.curr_batch_idx, self.memory_size
            )
        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = indexing.convert_to_pairs(
                    input_indices_tuple, labels
                )
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = indexing.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = indexing.concatenate_indices_tuples(
                indices_tuple, input_indices_tuple
            )
        return indices_tuple

    def reset_queue(self) -> None:
        self.embedding_memory.zero_()
        self.label_memory.zero_()
        self.has_been_filled = False
        self.queue_idx = 0
