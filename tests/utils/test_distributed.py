import logging
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import distributed

from .. import TEST_DEVICE, TEST_DTYPES, WITH_COLLECT_STATS


# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
def parameters_are_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        num_elements = float(torch.numel(p2.data))
        if torch.sum(torch.isclose(p1.data, p2.data, rtol=1e-2)) < (num_elements * 0.8):
            print("p1.data", p1.data)
            print("p2.data", p1.data)
            return False
    return True


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
    lr,
    model,
    inputs,
    labels,
    loss_fn,
    miner_fn,
    original_model,
    original_loss_fn,
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

    loss_fn = distributed.DistributedLossWrapper(loss=loss_fn)

    if miner_fn:
        miner_fn = distributed.DistributedMinerWrapper(miner=miner_fn)

    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=lr)
    optimizer.zero_grad()
    outputs = ddp_mp_model(inputs[rank].to(device))
    indices_tuple = None
    if miner_fn:
        indices_tuple = miner_fn(outputs, labels[rank])
    loss = loss_fn(outputs, labels[rank], indices_tuple)

    print(f"rank {rank} loss {loss}")

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


class TestDistributedLossWrapper(unittest.TestCase):
    def loss_and_miner_tester(self, loss_class, miner_class):
        torch.manual_seed(75210)
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
                original_loss_fn = loss_class(reducer=reducers.DoNothingReducer())
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

                print("all_outputs", all_outputs)
                indices_tuple = None
                if original_miner_fn:
                    indices_tuple = original_miner_fn(all_ouputs, all_labels)
                loss = original_loss_fn(all_outputs, all_labels, indices_tuple)
                print(
                    "TRUE len(losses[pos_loss][losses])",
                    len(loss["pos_loss"]["losses"]),
                )
                print(
                    "TRUE len(losses[neg_loss][losses])",
                    len(loss["neg_loss"]["losses"]),
                )
                loss1 = reducers.AvgNonZeroReducer()(
                    {"pos_loss": loss["pos_loss"]}, all_inputs, all_labels
                )
                loss2 = reducers.AvgNonZeroReducer()(
                    {"neg_loss": loss["neg_loss"]}, all_inputs, all_labels
                )
                loss = loss1 + loss2
                print("loss", loss)
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
                    ),
                    nprocs=world_size,
                    join=True,
                )

    def test_distributed_tuple_loss(self):
        self.loss_and_miner_tester(losses.ContrastiveLoss, None)

    # def test_distributed_tuple_loss_and_miner(self):
    #     self.loss_and_miner_tester(
    #         losses.ContrastiveLoss, miners.MultiSimilarityMiner
    #     )

    # def test_distributed_tuple_miner_with_ref_emb(self):
    #     self.loss_and_miner_tester(
    #         losses.ContrastiveLoss, miners.MultiSimilarityMiner, test_ref_emb=True
    #     )


if __name__ == "__main__":
    unittest.main()
