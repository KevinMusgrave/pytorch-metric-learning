import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import common_functions as c_f


# modified from https://github.com/allenai/allennlp
def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather(x):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    x_list = [torch.ones_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(x_list, x.contiguous())
    del x_list[rank]
    return torch.cat(x_list, dim=0)


# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather_embeddings_and_labels(embeddings, labels):
    labels = c_f.to_device(labels, embeddings)
    # If we are not using distributed training, this is a no-op.
    if not is_distributed():
        return None, None
    ref_emb = all_gather(embeddings)
    ref_labels = all_gather(labels)
    return ref_emb, ref_labels


def gather_and_concat(embeddings, labels, ref_emb, ref_labels):
    dist_ref_emb, dist_ref_labels = all_gather_embeddings_and_labels(embeddings, labels)
    if None not in [dist_ref_emb, ref_emb]:
        dist_ref_emb = torch.cat([dist_ref_emb, ref_emb], dim=0)
        dist_ref_labels = torch.cat([dist_ref_labels, ref_labels], dim=0)
    return dist_ref_emb, dist_ref_labels


class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(
        self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        dist_ref_emb, dist_ref_labels = gather_and_concat(
            embeddings, labels, ref_emb, ref_labels
        )
        loss = self.loss(
            embeddings, labels, indices_tuple, dist_ref_emb, dist_ref_labels
        )
        return loss * torch.distributed.get_world_size()


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner):
        super().__init__()
        self.miner = miner

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        dist_ref_emb, dist_ref_labels = gather_and_concat(
            embeddings, labels, ref_emb, ref_labels
        )
        return self.miner(embeddings, labels, dist_ref_emb, dist_ref_labels)
