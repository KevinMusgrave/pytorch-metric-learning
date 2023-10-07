import torch

from ..utils import common_functions as c_f
# replace the functions of loss_and_miner_utils by multisupcon's
from ..utils import multilabel_loss_and_miner_utils as mlmu
from ..utils import loss_and_miner_utils as lmu
from ..utils.module_with_records import ModuleWithRecords
from .base_loss_wrapper import BaseLossWrapper


class CrossBatchMemory4MultiLabel(BaseLossWrapper, ModuleWithRecords):
    def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.num_classes = loss.num_classes
        self.reset_queue()
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
        )

    @staticmethod
    def supported_losses():
        return [
            "MultiSupConLoss"
        ]

    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise Exception(f"CrossBatchMemory not supported for {loss_name}")

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_mask=None):
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")
        if enqueue_mask is not None:
            assert len(enqueue_mask) == len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )

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

        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory
        indices_tuple = self.create_indices_tuple(
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices_tuple,
            do_remove_self_comparisons,
        )
        loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
        return loss

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size
            )
            % self.memory_size
        )
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        # self.label_memory[self.curr_batch_idx] = labels
        for i in range(len(self.curr_batch_idx)):
            self.label_memory[self.curr_batch_idx[i]] = labels[i]
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def create_indices_tuple(
        self,
        embeddings,
        labels,
        E_mem,
        L_mem,
        input_indices_tuple,
        do_remove_self_comparisons,
    ):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = mlmu.get_all_pairs_indices(labels, self.num_classes, L_mem)
        if do_remove_self_comparisons:
            indices_tuple = lmu.remove_self_comparisons(
                indices_tuple, self.curr_batch_idx, self.memory_size
            )

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = mlmu.convert_to_pairs(input_indices_tuple, labels, self.num_classes)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = mlmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = c_f.concatenate_indices_tuples(
                indices_tuple, input_indices_tuple
            )

        return indices_tuple

    def reset_queue(self):
        self.register_buffer(
            "embedding_memory", torch.zeros(self.memory_size, self.embedding_size)
        )
        self.label_memory = [[] for i in range(self.memory_size)]
        self.has_been_filled = False
        self.queue_idx = 0
