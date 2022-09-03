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
    ref_labels = all_gather(labels)
    return ref_emb, ref_labels


def gather(emb, labels, ref_emb=None, ref_labels=None):
    device = emb.device
    labels = c_f.to_device(labels, device=device)
    rank = torch.distributed.get_rank()

    dist_emb, dist_labels = all_gather_embeddings_and_labels(emb, labels)
    all_emb = torch.cat([emb, dist_emb], dim=0)
    all_labels = torch.cat([labels, dist_labels], dim=0)

    if ref_emb != None and ref_labels != None:
        dist_ref_emb, dist_ref_labels = all_gather_embeddings_and_labels(
            ref_emb, ref_labels
        )
        all_ref_emb = torch.cat([ref_emb, dist_ref_emb], dim=0)
        all_ref_labels = torch.cat([ref_labels, dist_ref_labels], dim=0)
    else:
        all_ref_emb, all_ref_labels = None, None

    return all_emb, all_labels, all_ref_emb, all_ref_labels, labels, device


def get_indices_tuple(
    labels, ref_labels, device, embeddings=None, ref_emb=None, miner=None
):
    curr_batch_idx = torch.arange(len(labels), device=device)
    if miner:
        indices_tuple = miner(embeddings, labels, ref_emb, ref_labels)
    else:
        indices_tuple = lmu.get_all_pairs_indices(labels, ref_labels)
    return lmu.remove_self_comparisons(indices_tuple, curr_batch_idx, len(ref_labels))


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

    def forward(self, emb, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, ref_emb, ref_labels)

        all_emb, all_labels, all_ref_emb, all_ref_labels, labels, device = gather(
            emb, labels, ref_emb, ref_labels
        )

        if self.efficient:
            if indices_tuple is None:
                indices_tuple = get_indices_tuple(labels, all_labels, device)
            loss = self.loss(
                all_emb, all_labels, indices_tuple, all_ref_emb, all_ref_labels
            )
        else:
            loss = self.loss(
                all_emb, all_labels, indices_tuple, all_ref_emb, all_ref_labels
            )

        return loss * world_size


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner, efficient=False):
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError("The input miner must extend BaseMiner")
        self.miner = miner
        self.efficient = efficient

    def forward(self, emb, labels, ref_emb=None, ref_labels=None):
        all_emb, all_labels, all_ref_emb, all_ref_labels, labels, device = gather(
            emb, labels, ref_emb, ref_labels
        )
        if self.efficient:
            return get_indices_tuple(
                all_labels, all_ref_labels, device, all_emb, all_ref_emb, self.miner
            )
        else:
            return self.miner(all_emb, all_labels, all_ref_emb, all_ref_labels)
