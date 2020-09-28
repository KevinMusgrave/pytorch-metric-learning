import unittest
import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from pytorch_metric_learning.utils import distributed, common_functions as c_f
from pytorch_metric_learning import losses, miners
from torch.nn.parallel import DistributedDataParallel as DDP
from .. import TEST_DTYPES, TEST_DEVICE

# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
def parameters_are_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        num_elements = float(torch.numel(p2.data))
        if torch.sum(torch.isclose(p1.data, p2.data, rtol=1e-2)) < (num_elements * 0.8):
            return False
    return True


### from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html ###
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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
    original_miner_fn,
    correct_loss,
    correct_indices_tuple,
    is_tuple_loss,
    ref_outputs,
    ref_labels,
):
    setup(rank, world_size)
    device = torch.device("cuda:{}".format(rank))

    ddp_mp_model = DDP(model.to(device), device_ids=[rank], output_device=rank)

    if is_tuple_loss:
        loss_fn = distributed.DistributedLossWrapper(loss=loss_fn)
    else:
        loss_fn = distributed.DistributedLossWrapper(
            loss=loss_fn.to(device), device_ids=[rank], output_device=rank
        )
        loss_optimizer = optim.SGD(loss_fn.parameters(), lr=lr)
        loss_optimizer.zero_grad()

    miner_fn = distributed.DistributedMinerWrapper(miner=miner_fn)

    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=lr)
    optimizer.zero_grad()
    outputs = ddp_mp_model(inputs[rank].to(device))

    if ref_outputs is not None:
        ref_outputs[rank] = ref_outputs[rank].to(device)
        indices_tuple = miner_fn(
            outputs, labels[rank], ref_outputs[rank], ref_labels[rank]
        )
        indices_tuple = c_f.shift_indices_tuple(
            indices_tuple, len(outputs) * world_size
        )
        loss = loss_fn(
            [outputs, ref_outputs[rank]],
            [labels[rank], ref_labels[rank]],
            indices_tuple,
        )
    else:
        indices_tuple = miner_fn(outputs, labels[rank])
        loss = loss_fn(outputs, labels[rank], indices_tuple)

    if is_tuple_loss:
        pos_loss_size = loss_fn.loss.reducer.reducers["pos_loss"].losses_size
        neg_loss_size = loss_fn.loss.reducer.reducers["neg_loss"].losses_size
        correct_pos_loss_size = original_loss_fn.reducer.reducers[
            "pos_loss"
        ].losses_size
        correct_neg_loss_size = original_loss_fn.reducer.reducers[
            "neg_loss"
        ].losses_size
        assert pos_loss_size == correct_pos_loss_size
        assert neg_loss_size == correct_neg_loss_size
    else:
        loss_size = loss_fn.loss.module.reducer.losses_size
        correct_loss_size = original_loss_fn.reducer.losses_size
        assert loss_size == correct_loss_size

    assert torch.isclose(loss, torch.from_numpy(correct_loss).to(device))
    assert miner_fn.miner.num_pos_pairs == original_miner_fn.num_pos_pairs
    assert miner_fn.miner.num_neg_pairs == original_miner_fn.num_neg_pairs
    for i in range(len(correct_indices_tuple)):
        assert torch.all(
            indices_tuple[i] == (torch.from_numpy(correct_indices_tuple[i]).to(device))
        )

    dist.barrier()
    loss.backward()

    original_model = original_model.to(device)
    assert not parameters_are_equal(original_model, ddp_mp_model.module)
    dist.barrier()
    optimizer.step()
    dist.barrier()
    assert parameters_are_equal(original_model, ddp_mp_model.module)

    if not is_tuple_loss:
        original_loss_fn = original_loss_fn.to(device)
        assert not parameters_are_equal(original_loss_fn, loss_fn.loss.module)
        dist.barrier()
        loss_optimizer.step()
        dist.barrier()
        assert parameters_are_equal(original_loss_fn, loss_fn.loss.module)

    dist.barrier()
    cleanup()


class TestDistributedLossWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        TEST_DEVICE = torch.device("cuda")

    def create_loss(self, loss_class, is_tuple_loss, dtype):
        if is_tuple_loss:
            return loss_class()
        else:
            return loss_class(num_classes=2, embedding_size=5).type(dtype)

    def loss_and_miner_tester(
        self, loss_class, miner_class, is_tuple_loss, test_ref_emb=False
    ):
        for dtype in TEST_DTYPES:
            for world_size in range(1, 5):
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

                original_model = original_model.to(TEST_DEVICE)
                original_loss_fn = self.create_loss(loss_class, is_tuple_loss, dtype)
                loss_fn = self.create_loss(loss_class, is_tuple_loss, dtype)
                if not is_tuple_loss:
                    loss_fn.load_state_dict(original_loss_fn.state_dict())
                    assert parameters_are_equal(original_loss_fn, loss_fn)
                    original_loss_fn = original_loss_fn.to(TEST_DEVICE)
                    loss_optimizer = optim.SGD(original_loss_fn.parameters(), lr=lr)
                    loss_optimizer.zero_grad()

                original_miner_fn = miner_class()
                miner_fn = miner_class()

                optimizer = optim.SGD(original_model.parameters(), lr=lr)
                optimizer.zero_grad()
                all_inputs = torch.cat(inputs, dim=0).to(TEST_DEVICE)
                all_labels = torch.cat(labels, dim=0).to(TEST_DEVICE)
                all_outputs = original_model(all_inputs)
                if test_ref_emb:
                    ref_outputs = [
                        torch.randn(batch_size, 5).type(dtype).detach()
                        for _ in range(world_size)
                    ]
                    ref_labels = [
                        torch.randint(low=0, high=2, size=(batch_size,))
                        for _ in range(world_size)
                    ]
                    all_ref_outputs = torch.cat(ref_outputs, dim=0).to(TEST_DEVICE)
                    all_ref_labels = torch.cat(ref_labels, dim=0).to(TEST_DEVICE)
                    correct_indices_tuple = original_miner_fn(
                        all_outputs, all_labels, all_ref_outputs, all_ref_labels
                    )
                    correct_indices_tuple = c_f.shift_indices_tuple(
                        correct_indices_tuple, len(all_outputs)
                    )
                    all_outputs = torch.cat([all_outputs, all_ref_outputs], dim=0).to(
                        TEST_DEVICE
                    )
                    all_labels = torch.cat([all_labels, all_ref_labels], dim=0).to(
                        TEST_DEVICE
                    )
                else:
                    ref_outputs, ref_labels = None, None
                    correct_indices_tuple = original_miner_fn(all_outputs, all_labels)
                correct_loss = original_loss_fn(
                    all_outputs, all_labels, correct_indices_tuple
                )
                (correct_loss / world_size).backward(retain_graph=True)
                optimizer.step()
                if not is_tuple_loss:
                    for p in original_loss_fn.parameters():
                        # Each replica loss function sees gradients from the global batch
                        p.grad *= world_size
                    loss_optimizer.step()

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
                        original_miner_fn,
                        correct_loss.detach().cpu().numpy(),
                        tuple([x.cpu().numpy() for x in correct_indices_tuple]),
                        is_tuple_loss,
                        ref_outputs,
                        ref_labels,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    def test_distributed_tuple_loss_and_miner(self):
        self.loss_and_miner_tester(
            losses.ContrastiveLoss, miners.MultiSimilarityMiner, True
        )

    def test_distributed_classifier_loss_and_miner(self):
        self.loss_and_miner_tester(
            losses.ArcFaceLoss, miners.MultiSimilarityMiner, False
        )

    def test_distributed_tuple_miner_with_ref_emb(self):
        self.loss_and_miner_tester(
            losses.ContrastiveLoss, miners.MultiSimilarityMiner, True, test_ref_emb=True
        )


if __name__ == "__main__":
    unittest.main()
