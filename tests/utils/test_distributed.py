import logging
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import distributed

from .. import TEST_DEVICE, TEST_DTYPES


# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
def parameters_are_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        return torch.allclose(p1.data, p2.data, rtol=1e-2)
    return True


### from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html ###
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"

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
    lr,
    model,
    inputs,
    labels,
    loss_fn,
    miner_fn,
    original_model,
    original_loss_fn,
    efficient,
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

    loss_fn = distributed.DistributedLossWrapper(loss=loss_fn, efficient=efficient)

    if miner_fn:
        miner_fn = distributed.DistributedMinerWrapper(
            miner=miner_fn, efficient=efficient
        )

    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=lr)
    optimizer.zero_grad()
    outputs = ddp_mp_model(inputs[rank].to(device))
    indices_tuple = None
    if miner_fn:
        indices_tuple = miner_fn(outputs, labels[rank])
    loss = loss_fn(outputs, labels[rank], indices_tuple)

    dist.barrier()
    loss.backward()

    original_model = original_model.to(device)
    assert not parameters_are_equal(original_model, ddp_mp_model.module)
    dist.barrier()
    optimizer.step()
    dist.barrier()
    assert parameters_are_equal(original_model, ddp_mp_model.module)
    dist.barrier()
    cleanup()


def create_efficient_batch(x, i, batch_size):
    s = i * batch_size
    e = (i + 1) * batch_size
    curr = x[s:e]
    others = torch.cat([x[:s], x[e:]], dim=0).detach()
    others = torch.cat([curr, others], dim=0)
    return curr, others


class TestDistributedLossWrapper(unittest.TestCase):
    def loss_and_miner_tester(self, loss_class, miner_class, efficient):
        torch.manual_seed(75210)
        if TEST_DEVICE == torch.device("cpu"):
            return
        max_world_size = min(4, torch.cuda.device_count())
        if max_world_size < 1:
            logging.warning(
                "In GPU mode but no GPUs available. Skipping distributed test"
            )
            return
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            for world_size in range(2, max_world_size + 1):
                batch_size = 20
                lr = 1
                inputs = [
                    torch.randn(batch_size, 10).type(dtype) for _ in range(world_size)
                ]
                labels = [
                    torch.randint(low=0, high=2, size=(batch_size,))
                    for _ in range(world_size)
                ]
                original_model = ToyMpModel().type(dtype)
                model = ToyMpModel().type(dtype)
                model.load_state_dict(original_model.state_dict())
                self.assertTrue(parameters_are_equal(original_model, model))

                original_model = original_model.to(TEST_DEVICE)
                original_loss_fn = loss_class()
                loss_fn = loss_class()

                if miner_class:
                    original_miner_fn = miner_class()
                    miner_fn = miner_class()
                else:
                    original_miner_fn = None
                    miner_fn = None

                optimizer = optim.SGD(original_model.parameters(), lr=lr)
                optimizer.zero_grad()
                all_inputs = torch.cat(inputs, dim=0).to(TEST_DEVICE)
                all_labels = torch.cat(labels, dim=0).to(TEST_DEVICE)
                all_outputs = original_model(all_inputs)
                indices_tuple = None
                if efficient:
                    losses = []
                    for i in range(len(inputs)):
                        curr_emb, other_emb = create_efficient_batch(
                            all_outputs, i, batch_size
                        )
                        curr_labels, other_labels = create_efficient_batch(
                            all_labels, i, batch_size
                        )
                        if original_miner_fn:
                            indices_tuple = distributed.get_indices_tuple(
                                curr_labels,
                                other_labels,
                                TEST_DEVICE,
                                embeddings=curr_emb,
                                ref_emb=other_emb,
                                miner=original_miner_fn,
                            )
                        else:
                            indices_tuple = distributed.get_indices_tuple(
                                curr_labels, other_labels, TEST_DEVICE
                            )
                        loss = original_loss_fn(
                            curr_emb,
                            curr_labels,
                            indices_tuple,
                            other_emb,
                            other_labels,
                        )
                        losses.append(loss)
                    loss = sum(losses)
                else:
                    if original_miner_fn:
                        indices_tuple = original_miner_fn(all_outputs, all_labels)
                    loss = original_loss_fn(all_outputs, all_labels, indices_tuple)
                loss.backward()
                optimizer.step()

                mp.spawn(
                    single_process_function,
                    args=(
                        world_size,
                        lr,
                        model,
                        inputs,
                        labels,
                        loss_fn,
                        miner_fn,
                        original_model,
                        original_loss_fn,
                        efficient,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    def test_distributed_tuple_loss(self):
        self.loss_and_miner_tester(losses.ContrastiveLoss, None, False)

    def test_distributed_tuple_loss_and_miner(self):
        self.loss_and_miner_tester(
            losses.ContrastiveLoss, miners.MultiSimilarityMiner, False
        )

    def test_distributed_tuple_loss_efficient(self):
        self.loss_and_miner_tester(losses.ContrastiveLoss, None, True)

    def test_distributed_tuple_loss_and_miner_efficient(self):
        self.loss_and_miner_tester(
            losses.ContrastiveLoss, miners.MultiSimilarityMiner, True
        )


if __name__ == "__main__":
    unittest.main()
