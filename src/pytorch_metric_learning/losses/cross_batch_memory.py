import torch

from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from ..utils.module_with_records import ModuleWithRecords


class CrossBatchMemory(ModuleWithRecords):
    def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.embedding_memory = torch.zeros(self.memory_size, self.embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
        )

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_idx=None):
        if enqueue_idx is not None:
            assert len(enqueue_idx) <= len(self.embedding_memory)
            assert len(enqueue_idx) < len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )

        if enqueue_idx is not None:
            mask = torch.zeros(len(embeddings), device=device, dtype=torch.bool)
            mask[enqueue_idx] = True
            emb_for_queue = embeddings[mask]
            labels_for_queue = labels[mask]
            embeddings = embeddings[~mask]
            labels = labels[~mask]
            do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True

        batch_size = len(embeddings)
        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory

        indices_tuple = self.create_indices_tuple(
            batch_size,
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices_tuple,
            do_remove_self_comparisons,
        )
        combined_embeddings = torch.cat([embeddings, E_mem], dim=0)
        combined_labels = torch.cat([labels, L_mem], dim=0)
        loss = self.loss(combined_embeddings, combined_labels, indices_tuple)
        return loss

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def create_indices_tuple(
        self,
        batch_size,
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
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)

        if do_remove_self_comparisons:
            indices_tuple = self.remove_self_comparisons(indices_tuple)

        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = tuple(
                [
                    torch.cat([x, c_f.to_device(y, x)], dim=0)
                    for x, y in zip(indices_tuple, input_indices_tuple)
                ]
            )

        return indices_tuple

    def remove_self_comparisons(self, indices_tuple):
        # remove self-comparisons
        assert len(indices_tuple) in [3, 4]
        s, e = self.curr_batch_idx[0], self.curr_batch_idx[-1]
        if len(indices_tuple) == 3:
            a, p, n = indices_tuple
            keep_mask = self.not_self_comparisons(a, p, s, e)
            a = a[keep_mask]
            p = p[keep_mask]
            n = n[keep_mask]
            assert len(a) == len(p) == len(n)
            return a, p, n
        elif len(indices_tuple) == 4:
            a1, p, a2, n = indices_tuple
            keep_mask = self.not_self_comparisons(a1, p, s, e)
            a1 = a1[keep_mask]
            p = p[keep_mask]
            assert len(a1) == len(p)
            assert len(a2) == len(n)
            return a1, p, a2, n

    # a: anchors
    # p: positives
    # s: curr batch start idx in queue
    # e: curr batch end idx in queue
    def not_self_comparisons(self, a, p, s, e):
        curr_batch = torch.any(p.unsqueeze(1) == self.curr_batch_idx, dim=1)
        a_c = a[curr_batch]
        p_c = p[curr_batch]
        p_c -= s
        if e <= s:
            p_c[p_c <= e - s] += self.memory_size
        without_self_comparisons = curr_batch.clone()
        without_self_comparisons[torch.where(curr_batch)[0][a_c == p_c]] = False
        return without_self_comparisons | ~curr_batch
