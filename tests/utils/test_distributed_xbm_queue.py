import unittest
from .. import TEST_DEVICE, TEST_DTYPES


import os
import argparse
import random
import warnings
import numpy as np


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import distributed as pml_dist
# def _init_fn(args):
#     def func(worker_id):
#         SEED=args.seed + worker_id
#         torch.manual_seed(SEED)
#         torch.cuda.manual_seed(SEED)
#         np.random.seed(SEED)
#         random.seed(SEED)
#         torch.backends.cudnn.deterministic=True
#         torch.backends.cudnn.benchmark = False
#     return func

def _init_fn(worker_id):
    SEED = worker_id
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def example(local_rank, rank, world_size, args):
    print(f"current local_rank is {local_rank}, current rank is {rank}")


    # ============================DDP specific ===================================#
    # create default process group
    dist.init_process_group("nccl")# , init_method='env://')#, rank=rank, world_size=world_size)   # rank=rank, world_size=world_size) # not needed since aml launch scripts have the EVN VAR set already?

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # dataset
    dataset = RandomDataset(args.sample_size)

    print(f"world size is {world_size}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        drop_last=True,
        seed = args.seed#,
        # num_replicas=world_size,  # not needed since aml launch scripts have the EVN VAR set already?
        # rank=rank
    )
    # ============================DDP specific ===================================#
    args.batch_size //= world_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers_per_process, pin_memory=True, sampler=train_sampler,  worker_init_fn=_init_fn)

    # create local model
    model_q = nn.Linear(10, 10).to(local_rank)
    with torch.no_grad():
        model_k = nn.Linear(10, 10).to(local_rank)

    #============================DDP specific end ===================================#
    # construct DDP model
    model_q = DDP(model_q, device_ids=[local_rank])

    # define loss function and optimizer
    # miner = miners.MultiSimilarityMiner()
    # set margins to ensure that no pairs are left out for this example
    miner = miners.PairMarginMiner(pos_margin=0, neg_margin=100)
    # miner = pml_dist.DistributedMinerWrapper(miner=miner, efficient=True) ## miner is encoded inside loss in our case, then the miner will have all gathered embs, so no wrapper is needed
    loss_fn = losses.CrossBatchMemory(loss=losses.NTXentLoss(temperature=0.07), embedding_size=10,
                                      memory_size=args.memory_size, miner=miner)
    loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn)
    # ============================DDP specific end ===================================#
    optimizer = optim.SGD(model_q.parameters(), lr=0.001)


    ## train loop
    # Iterate over the data using the DataLoader
    for epoch in range(args.num_of_epoch):
        train_sampler.set_epoch(epoch)
        for worker_id, index, batch_inputs, batch_outputs in dataloader:
            print(f"in epoch {epoch}, worker id {worker_id}, index {index}, label {batch_outputs}")
            # forward pass
            embeds_Q = model_q(batch_inputs.to(local_rank))
            labels = batch_outputs.to(local_rank)

            # compute output
            with torch.no_grad():  # no gradient to keys
                copy_params(model_q, model_k, m=0.99)
                # ============================DDP specific ===================================#
                # shuffle for making use of BN
                imgK, idx_unshuffle = batch_shuffle_ddp(batch_inputs.to(local_rank))
                embeds_K = model_k(imgK)
                embeds_K = batch_unshuffle_ddp(embeds_K, idx_unshuffle)
                # ============================DDP specific end ===================================#

                # ========================== for debug ==================================#
                # print(f"input before shuffle on rank {rank}: {batch_inputs}")
                # print(f"input after shuffle on rank {rank}: {imgK}")
                gpu_idx = torch.distributed.get_rank()

                num_gpus = world_size
                idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

                # ========================== end for debug ==================================#
            ## same as original MOCO: same image augmented with the same label
            all_enc = torch.cat([embeds_Q, embeds_K], dim=0)
            labels, enqueue_mask = create_labels(embeds_Q.size(0), labels, local_rank)
            # # ========================= debug =======================#
            # print("======================== all gathering ====================")
            # world_size = torch.distributed.get_world_size()
            # print(f"world_size is {world_size}")
            # all_enc, labels, _, _, _ = pml_dist.gather_emb_and_ref(
            #     all_enc, labels
            # )
            # enqueue_mask = pml_dist.gather_enqueue_mask(enqueue_mask, all_enc.device)
            # # ========================= end of debug =======================#
            loss = loss_fn(all_enc, labels,
                           enqueue_mask=enqueue_mask)  # miner will be used in loss_fn if initialized with miner

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # check pair size
            print(f'label on mem bank on rank{rank}: {loss_fn.loss.label_memory}') # here we should see the same labels on mem queue on different workers
            print(f"bs is {batch_inputs.shape[0]}")
            print("num of pos pairs: ", loss_fn.loss.miner.num_pos_pairs)
            print("num of neg pairs: ", loss_fn.loss.miner.num_neg_pairs)
            print(f"epoch {epoch}, loss is {loss}")
    # print weights
    for param in model_q.parameters():
        print(param.data)

    dist.destroy_process_group()


def copy_params(encQ, encK, m=None):
    if m is None:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    else:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

class RandomDataset(Dataset):
    def __init__(self, sample_size):
        self.samples = torch.randn(sample_size, 10)
        # self.samples = torch.ones_like(self.samples)
        self.labels = torch.arange(sample_size)
        print(f'data set labels: {self.labels}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if not worker_info:
            raise NotImplementedError('Not implemented for num_workers=0')
        print(f"in worker {worker_info.id}, index {index}, label is {self.labels[index]}")
        return worker_info.id, index, self.samples[index], self.labels[index]
def create_labels(num_pos_pairs, labels, device):
    # create labels that indicate what the positive pairs are
    labels = torch.cat((labels, labels)).to(device)

    # we want to enqueue the output of encK, which is the 2nd half of the batch
    enqueue_mask = torch.zeros(len(labels)).bool()
    enqueue_mask[num_pos_pairs:] = True
    return labels, enqueue_mask
@torch.no_grad()
def batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    *** x should be on local_rank device
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).to(x.device)# .cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle

@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class TestDistributedXbmQueue(unittest.TestCase):
    def main(self):
        # Create the argument parser
        parser = argparse.ArgumentParser(description='Train a metric learning model for MRI protocol classification.')

        # Add arguments to the parser
        parser.add_argument('--sample_size', type=int,
                            default=32,
                            help='batch_size')
        parser.add_argument('--batch_size', type=int,
                            default=32,
                            help='global batch_size')
        parser.add_argument('--seed', type=int,
                            default=None,
                            help='world_size')
        parser.add_argument('--memory_size', type=int,
                            default=32,
                            help='memory_size')
        parser.add_argument('--num_workers_per_process', type=int,
                            default=8,
                            help='num_workers')
        parser.add_argument('--num_of_epoch', type=int,
                            default=5,
                            help='num_of_epoch')

        args = parser.parse_args()
        #########################################################
        # Get PyTorch environment variables for distributed training.
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # here, why no use of mp.spawn?  ==> aml submit script is supposed to take care of the EVN VARs:
        # MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE
        # RANK, LOCAL_RANK
        os.environ['NCCL_DEBUG'] = 'INFO'
        print(f"args.batch_size is {args.batch_size}, world_size is {world_size}")
        assert (args.batch_size % world_size) == 0
        example(local_rank, rank, world_size, args)


if __name__=="__main__":
    unittest.main()