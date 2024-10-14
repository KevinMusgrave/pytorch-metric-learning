import warnings

import torch

from ..losses import BaseMetricLossFunction, CrossBatchMemory
from ..miners import BaseMiner
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu


# modified from https://github.com/allenai/allennlp
def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather(x):
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        rank = torch.distributed.get_rank()
        x_list = [torch.ones_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x.contiguous())
        # remove curr rank
        x_list.pop(rank)
        return torch.cat(x_list, dim=0)
    return None


# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather_embeddings_and_labels(emb, labels):
    # If we are not using distributed training, this is a no-op.
    if not is_distributed():
        return None, None
    ref_emb = all_gather(emb)
    ref_labels = all_gather(labels) if labels is not None else None
    return ref_emb, ref_labels


def gather(emb, labels):
    device = emb.device
    if labels is not None:
        labels = c_f.to_device(labels, device=device)
    dist_emb, dist_labels = all_gather_embeddings_and_labels(emb, labels)
    all_emb = torch.cat([emb, dist_emb], dim=0)
    all_labels = (
        torch.cat([labels, dist_labels], dim=0) if dist_labels is not None else None
    )
    return all_emb, all_labels, labels


def gather_emb_and_ref(emb, labels, ref_emb=None, ref_labels=None):
    all_emb, all_labels, labels = gather(emb, labels)
    all_ref_emb, all_ref_labels = None, None

    if ref_emb is not None:
        all_ref_emb, all_ref_labels, _ = gather(ref_emb, ref_labels)

    return all_emb, all_labels, all_ref_emb, all_ref_labels, labels


def get_indices_tuple(labels, ref_labels, embeddings=None, ref_emb=None, miner=None):
    device = labels.device
    curr_batch_idx = torch.arange(len(labels), device=device)
    if miner:
        indices_tuple = miner(embeddings, labels, ref_emb, ref_labels)
    else:
        indices_tuple = lmu.get_all_pairs_indices(labels, ref_labels)
    return lmu.remove_self_comparisons(indices_tuple, curr_batch_idx, len(ref_labels))


def gather_enqueue_mask(enqueue_mask, device):
    if enqueue_mask is None:
        return enqueue_mask
    enqueue_mask = c_f.to_device(enqueue_mask, device=device)
    return torch.cat([enqueue_mask, all_gather(enqueue_mask)], dim=0)


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
        if not is_distributed():
            warnings.warn(
                "DistributedLossWrapper is being used in a non-distributed setting. Returning the loss as is."
            )
            return self.loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)

        world_size = torch.distributed.get_world_size()
        common_args = [
            embeddings,
            labels,
            indices_tuple,
            ref_emb,
            ref_labels,
            world_size,
        ]
        if isinstance(self.loss, CrossBatchMemory):
            return self.forward_cross_batch(*common_args, enqueue_mask)
        return self.forward_regular_loss(*common_args)

    def forward_regular_loss(
        self, emb, labels, indices_tuple, ref_emb, ref_labels, world_size
    ):
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, ref_emb, ref_labels)

        all_emb, all_labels, all_ref_emb, all_ref_labels, labels = gather_emb_and_ref(
            emb, labels, ref_emb, ref_labels
        )

        if self.efficient:
            if all_labels is not None:
                all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            if indices_tuple is None:
                indices_tuple = get_indices_tuple(labels, all_labels)
            loss = self.loss(emb, labels, indices_tuple, all_emb, all_labels)
        else:
            loss = self.loss(
                all_emb, all_labels, indices_tuple, all_ref_emb, all_ref_labels
            )

        return loss * world_size

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

        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, enqueue_mask)

        all_emb, all_labels, _, _, _ = gather_emb_and_ref(
            emb, labels, ref_emb, ref_labels
        )
        enqueue_mask = gather_enqueue_mask(enqueue_mask, emb.device)
        loss = self.loss(all_emb, all_labels, indices_tuple, enqueue_mask)
        return loss * world_size


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner, efficient=False):
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError("The input miner must extend BaseMiner")
        self.miner = miner
        self.efficient = efficient

    def forward(self, emb, labels, ref_emb=None, ref_labels=None):
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return self.miner(emb, labels, ref_emb, ref_labels)

        all_emb, all_labels, all_ref_emb, all_ref_labels, labels = gather_emb_and_ref(
            emb, labels, ref_emb, ref_labels
        )
        if self.efficient:
            all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            return get_indices_tuple(labels, all_labels, emb, all_emb, self.miner)
        else:
            return self.miner(all_emb, all_labels, all_ref_emb, all_ref_labels)
