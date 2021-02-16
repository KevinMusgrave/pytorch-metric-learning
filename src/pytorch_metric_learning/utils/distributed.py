import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu


# modified from https://github.com/allenai/allennlp
def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather(embeddings, labels, indices_tuple=None, get_indices_tuple=True):
    labels = c_f.to_device(labels, embeddings)
    # If we are not using distributed training, this is a no-op.
    if not is_distributed():
        return embeddings, labels
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    # Gather the embeddings on all replicas
    embeddings_list = [torch.ones_like(embeddings) for _ in range(world_size)]
    labels_list = [torch.ones_like(labels) for _ in range(world_size)]
    torch.distributed.all_gather(embeddings_list, embeddings.contiguous())
    torch.distributed.all_gather(labels_list, labels.contiguous())
    # The gathered copy of the current replicas embeddings have no gradients, so we overwrite
    # them with the embeddings generated on this replica, which DO have gradients.
    embeddings_list[rank] = embeddings
    labels_list[rank] = labels
    # Finally, we concatenate the embeddings
    all_embeddings = torch.cat(embeddings_list)
    all_labels = torch.cat(labels_list)

    if get_indices_tuple and indices_tuple is None:
        indices_tuple = lmu.get_all_pairs_indices(labels, all_labels)
    return all_embeddings, all_labels, indices_tuple


def all_gather_embeddings_labels(
    embeddings, labels, indices_tuple=None, get_indices_tuple=True
):
    if c_f.is_list_or_tuple(embeddings):
        assert c_f.is_list_or_tuple(labels)
        all_embeddings, all_labels = [], []
        for i in range(len(embeddings)):
            E, L = all_gather(embeddings[i], labels[i])
            all_embeddings.append(E)
            all_labels.append(L)
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
    else:
        embeddings, labels, indices_tuple = all_gather(
            embeddings, labels, indices_tuple, get_indices_tuple
        )

    return embeddings, labels, indices_tuple


class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss, **kwargs):
        super().__init__()
        has_parameters = len([p for p in loss.parameters()]) > 0
        self.loss = DDP(loss, **kwargs) if has_parameters else loss

    def forward(self, embeddings, labels, indices_tuple=None, *args, **kwargs):
        embeddings, labels, indices_tuple = all_gather_embeddings_labels(
            embeddings, labels, indices_tuple
        )
        return self.loss(embeddings, labels, indices_tuple, **kwargs)


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner):
        super().__init__()
        self.miner = miner

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        if ref_emb is not None:
            ref_emb, ref_labels, _ = all_gather_embeddings_labels(
                ref_emb, ref_labels, get_indices_tuple=False
            )
        else:
            ref_emb, ref_labels, _ = all_gather_embeddings_labels(
                embeddings, labels, get_indices_tuple=False
            )
        return self.miner(embeddings, labels, ref_emb, ref_labels)
