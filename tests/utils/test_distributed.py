import logging
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import distributed
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES


### from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html ###
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist_type = "gloo" if TEST_DEVICE == torch.device("cpu") else "nccl"
    # initialize the process group
    dist.init_process_group(dist_type, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyMpModel(torch.nn.Module):
    def __init__(self):
        super(ToyMpModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


### from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html ###
def single_process_function(
    rank,
    world_size,
    model,
    inputs,
    labels,
    loss_fn,
    miner_fn,
    correct_losses,
    correct_indices_tuple,
    is_tuple_loss,
):
    setup(rank, world_size)
    if TEST_DEVICE == torch.device("cpu"):
        device = TEST_DEVICE
        device_ids = None
        output_device = None
    else:
        device = torch.device("cuda:{}".format(rank))
        device_ids = [rank]
        output_device = rank

    ddp_mp_model = DDP(
        model.to(device), device_ids=device_ids, output_device=output_device
    )

    if is_tuple_loss:
        loss_fn = distributed.DistributedLossWrapper(loss=loss_fn)
    else:
        loss_fn = distributed.DistributedLossWrapper(
            loss=loss_fn.to(device), device_ids=device_ids, output_device=output_device
        )

    outputs = ddp_mp_model(inputs[rank].to(device))

    if miner_fn is not None:
        miner_fn = distributed.DistributedMinerWrapper(miner=miner_fn)
        indices_tuple = miner_fn(outputs, labels[rank])
        loss = loss_fn(outputs, labels[rank], indices_tuple)
        for i in range(len(correct_indices_tuple[rank])):
            assert torch.all(
                indices_tuple[i]
                == (torch.from_numpy(correct_indices_tuple[rank][i]).to(device))
            )
    else:
        loss = loss_fn(outputs, labels[rank])

    assert torch.isclose(loss, torch.from_numpy(correct_losses[rank]).to(device))

    dist.barrier()
    cleanup()


class TestDistributedLossWrapper(unittest.TestCase):
    def create_loss(self, loss_class, is_tuple_loss, dtype, num_classes):
        if is_tuple_loss:
            return loss_class()
        else:
            return loss_class(num_classes=num_classes, embedding_size=5).type(dtype)

    def loss_and_miner_tester(self, loss_class, miner_class, is_tuple_loss):
        num_classes = 3
        if TEST_DEVICE != torch.device("cpu"):
            max_world_size = min(4, torch.cuda.device_count())
            if max_world_size < 1:
                logging.warning(
                    "In GPU mode but no GPUs available. Skipping distributed test"
                )
                return
        else:
            max_world_size = 2
        for dtype in TEST_DTYPES:
            for world_size in range(1, max_world_size + 1):
                batch_size = 32
                lr = 1
                inputs = [
                    torch.randn(batch_size, 10).type(dtype).to(TEST_DEVICE)
                    for _ in range(world_size)
                ]
                labels = [
                    torch.randint(low=0, high=num_classes, size=(batch_size,)).to(
                        TEST_DEVICE
                    )
                    for _ in range(world_size)
                ]
                original_model = ToyMpModel().type(dtype)
                model = ToyMpModel().type(dtype)
                model.load_state_dict(original_model.state_dict())

                original_model = original_model.to(TEST_DEVICE)
                original_loss_fn = self.create_loss(
                    loss_class, is_tuple_loss, dtype, num_classes
                )
                loss_fn = self.create_loss(
                    loss_class, is_tuple_loss, dtype, num_classes
                )
                if not is_tuple_loss:
                    loss_fn.load_state_dict(original_loss_fn.state_dict())
                    assert parameters_are_equal(original_loss_fn, loss_fn)
                    original_loss_fn = original_loss_fn.to(TEST_DEVICE)
                    loss_optimizer = optim.SGD(original_loss_fn.parameters(), lr=lr)
                    loss_optimizer.zero_grad()

                all_labels = torch.cat(labels, dim=0).to(TEST_DEVICE)
                outputs = [original_model(x).to(TEST_DEVICE) for x in inputs]
                all_outputs = torch.cat(outputs, dim=0)
                if miner_class is not None:
                    original_miner_fn = miner_class()
                    miner_fn = miner_class()
                    correct_indices_tuple = [
                        original_miner_fn(x, y, all_outputs, all_labels)
                        for (x, y) in zip(outputs, labels)
                    ]
                else:
                    miner_fn = None
                    correct_indices_tuple = [
                        lmu.get_all_pairs_indices(y, all_labels) for y in labels
                    ]

                correct_losses = []

                for i in range(len(outputs)):
                    correct_losses.append(
                        original_loss_fn(
                            all_outputs, all_labels, correct_indices_tuple[i]
                        )
                    )

                correct_losses = [x.detach().cpu().numpy() for x in correct_losses]

                mp.spawn(
                    single_process_function,
                    args=(
                        world_size,
                        model,
                        inputs,
                        labels,
                        loss_fn,
                        miner_fn,
                        correct_losses,
                        [
                            tuple([x.cpu().numpy() for x in y])
                            for y in correct_indices_tuple
                        ],
                        is_tuple_loss,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    def test_distributed_tuple_loss_and_miner(self):
        self.loss_and_miner_tester(
            losses.ContrastiveLoss, miner_class=None, is_tuple_loss=True
        )

        self.loss_and_miner_tester(
            losses.ContrastiveLoss,
            miner_class=miners.MultiSimilarityMiner,
            is_tuple_loss=True,
        )

    # def test_distributed_classifier_loss_and_miner(self):
    #     self.loss_and_miner_tester(
    #         losses.ArcFaceLoss, miners.MultiSimilarityMiner, False
    #     )


if __name__ == "__main__":
    unittest.main()
