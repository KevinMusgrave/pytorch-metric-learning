import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..losses import BaseMetricLossFunction
from ..miners import BaseMiner
from ..reducers import DoNothingReducer
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
def all_gather_embeddings_and_labels(embeddings, labels):
    # If we are not using distributed training, this is a no-op.
    if not is_distributed():
        return None, None
    ref_emb = all_gather(embeddings)
    ref_labels = all_gather(labels)
    return ref_emb, ref_labels


def gather_and_concat(embeddings, labels, ref_emb, ref_labels):
    dist_ref_emb, dist_ref_labels = all_gather_embeddings_and_labels(
        embeddings, labels
    )
    if None not in [dist_ref_emb, ref_emb]:
        dist_ref_emb = torch.cat([dist_ref_emb, ref_emb], dim=0)
        dist_ref_labels = torch.cat([dist_ref_labels, ref_labels], dim=0)
    return dist_ref_emb, dist_ref_labels


class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        if not isinstance(loss, BaseMetricLossFunction):
            raise TypeError("The input loss must extend BaseMetricLossFunction")
        self.loss = loss
        self.reducer = self.loss.reducer
        self.loss.reducer = DoNothingReducer()

    def forward(
        self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if world_size > 1:
            dist_ref_emb, dist_ref_labels = gather_and_concat(
                embeddings, labels, ref_emb, ref_labels
            )
            if indices_tuple is not None:
                raise ValueError("indices_tuple not supported yet")
            
            curr_batch_idx = torch.arange(len(embeddings), device=device)
            curr_dist_ref_emb = torch.cat([embeddings, dist_ref_emb], dim=0)
            curr_dist_ref_labels = torch.cat([labels, dist_ref_labels], dim=0)
            indices_tuple = self.get_indices_tuple(
                labels, curr_dist_ref_labels, curr_batch_idx
            )
            losses = self.loss(
                embeddings, labels, indices_tuple, curr_dist_ref_emb, curr_dist_ref_labels
            )
            print(f"rank {rank} len(losses[pos_loss][losses])", len(losses["pos_loss"]["losses"]))
            print(f"rank {rank} len(losses[neg_loss][losses])", len(losses["neg_loss"]["losses"]))

            # losses2 = self.loss(
            #     dist_ref_emb, dist_ref_labels, ref_emb=embeddings, ref_labels=labels
            # )
            # print(f"rank {rank} len(losses2[pos_loss][losses])", len(losses2["pos_loss"]["losses"]))
            # print(f"rank {rank} len(losses2[neg_loss][losses])", len(losses2["neg_loss"]["losses"]))
            # c_f.merge_loss_dicts(losses, losses2)
            # print(f"rank {rank} len(final[pos_loss][losses])", len(losses["pos_loss"]["losses"]))
            # print(f"rank {rank} len(final[neg_loss][losses])", len(losses["neg_loss"]["losses"]))

            loss = self.reducer(losses, embeddings, labels)
        else:
            losses = self.loss(embeddings, labels, indices_tuple)
            loss = self.reducer(losses, embeddings, labels)
        return loss

    def get_indices_tuple(self, labels, ref_labels, curr_batch_idx, ref_is_subset=False):
        indices_tuple = lmu.get_all_pairs_indices(labels, ref_labels)
        return lmu.remove_self_comparisons(
            indices_tuple, curr_batch_idx, len(ref_labels), ref_is_subset
        )


class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner):
        super().__init__()
        if not isinstance(loss, BaseMiner):
            raise TypeError("The input miner must extend BaseMiner")
        self.miner = miner

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        dist_ref_emb, dist_ref_labels = gather_and_concat(
            embeddings, labels, ref_emb, ref_labels
        )
        return self.miner(embeddings, labels, dist_ref_emb, dist_ref_labels)
