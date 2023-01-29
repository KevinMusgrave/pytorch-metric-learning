import logging
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.miners import PairMarginMiner
from pytorch_metric_learning.utils import distributed
from pytorch_metric_learning.wrappers import CrossBatchMemory

from .. import TEST_DEVICE, TEST_DTYPES


# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
def parameters_are_equal(model1, model2):
    output = True
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        output &= torch.allclose(p1.data, p2.data, rtol=1e-2)
    return output


### from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html ###
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"

    dist_type = "gloo" if TEST_DEVICE == torch.device("cpu") else "nccl"
    # initialize the process group
    dist.init_process_group(dist_type, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyMpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
    iterations,
    model,
    inputs,
    labels,
    ref_inputs,
    ref_labels,
    loss_fn,
    miner_fn,
    original_model,
    efficient,
    pass_labels_to_loss_fn,
    use_xbm_enqueue_idx,
    enqueue_idx,
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

    original_model = original_model.to(device)
    assert not parameters_are_equal(original_model, ddp_mp_model.module)

    for i in range(iterations):
        optimizer.zero_grad()
        outputs = ddp_mp_model(inputs[i][rank].to(device))
        curr_labels = labels[i][rank]
        ref_outputs, curr_ref_labels = None, None
        if ref_inputs:
            ref_outputs = ddp_mp_model(ref_inputs[i][rank].to(device))
            curr_ref_labels = ref_labels[i][rank]
        indices_tuple = None
        if miner_fn:
            indices_tuple = miner_fn(outputs, curr_labels, ref_outputs, curr_ref_labels)
        if miner_fn and not pass_labels_to_loss_fn:
            loss = loss_fn(outputs, indices_tuple=indices_tuple, ref_emb=ref_outputs)
        elif use_xbm_enqueue_idx and isinstance(loss_fn.loss, CrossBatchMemory):
            loss = loss_fn(
                outputs, curr_labels, indices_tuple, enqueue_idx=enqueue_idx[rank]
            )
        else:
            loss = loss_fn(
                outputs, curr_labels, indices_tuple, ref_outputs, curr_ref_labels
            )

        dist.barrier()
        loss.backward()
        dist.barrier()
        optimizer.step()

    dist.barrier()
    assert parameters_are_equal(original_model, ddp_mp_model.module)
    dist.barrier()
    cleanup()


def create_efficient_batch(all_outputs, all_ref_outputs, i, batch_size):
    curr_source = all_outputs
    other_source = all_outputs if all_ref_outputs is None else all_ref_outputs
    s = i * batch_size
    e = (i + 1) * batch_size
    curr = curr_source[s:e]
    others = torch.cat([other_source[:s], other_source[e:]], dim=0).detach()
    if all_ref_outputs is None:
        others = torch.cat([curr, others], dim=0)
    else:
        others = torch.cat([other_source[s:e], others], dim=0)
    return curr, others


def create_inputs(batch_size, world_size, iterations, dtype):
    return [
        [torch.randn(batch_size, 10).type(dtype) for _ in range(world_size)]
        for _ in range(iterations)
    ]


def create_labels(batch_size, world_size, iterations):
    return [
        [torch.randint(low=0, high=2, size=(batch_size,)) for _ in range(world_size)]
        for _ in range(iterations)
    ]


def create_enqueue_idx(batch_size, world_size):
    # enqueue every other embedding
    local_enqueue_idx = [
        (torch.randint(0, batch_size, size=(batch_size // 4,))).long()
        for _ in range(world_size)
    ]
    global_enqueue_idx = []
    for i, x in enumerate(local_enqueue_idx):
        if i == 0:
            global_enqueue_idx.append(x)
        else:
            global_enqueue_idx.append(x + batch_size)
    global_enqueue_idx = torch.cat(global_enqueue_idx, dim=0)
    return local_enqueue_idx, global_enqueue_idx


def get_all_outputs_and_labels(inputs, labels, model, iteration):
    all_inputs = torch.cat(inputs[iteration], dim=0).to(TEST_DEVICE)
    all_labels = torch.cat(labels[iteration], dim=0).to(TEST_DEVICE)
    all_outputs = model(all_inputs)
    return all_outputs, all_labels


class TestDistributedLossWrapper(unittest.TestCase):
    def loss_and_miner_tester(
        self,
        loss_class,
        miner_class,
        efficient,
        xbm,
        use_ref,
        loss_kwargs=None,
        miner_kwargs=None,
        pass_labels_to_loss_fn=True,
        use_xbm_enqueue_idx=False,
    ):
        torch.manual_seed(75210)
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        miner_kwargs = {} if miner_kwargs is None else miner_kwargs
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
                lr = 0.1
                iterations = 10
                original_model = ToyMpModel().type(dtype)
                model = ToyMpModel().type(dtype)
                model.load_state_dict(original_model.state_dict())
                self.assertTrue(parameters_are_equal(original_model, model))

                original_model = original_model.to(TEST_DEVICE)
                original_loss_fn = loss_class(**loss_kwargs)
                loss_fn = loss_class(**loss_kwargs)
                if xbm:
                    original_loss_fn = CrossBatchMemory(
                        original_loss_fn, embedding_size=5
                    )
                    loss_fn = CrossBatchMemory(loss_fn, embedding_size=5)

                if miner_class:
                    original_miner_fn = miner_class(**miner_kwargs)
                    miner_fn = miner_class(**miner_kwargs)
                else:
                    original_miner_fn = None
                    miner_fn = None

                optimizer = optim.SGD(original_model.parameters(), lr=lr)
                inputs = create_inputs(batch_size, world_size, iterations, dtype)
                labels = create_labels(batch_size, world_size, iterations)
                ref_inputs, ref_labels, all_ref_outputs, all_ref_labels = (
                    None,
                    None,
                    None,
                    None,
                )
                if use_ref:
                    ref_inputs = create_inputs(
                        batch_size, world_size, iterations, dtype
                    )
                    ref_labels = create_labels(batch_size, world_size, iterations)

                local_enqueue_idx, global_enqueue_idx = create_enqueue_idx(
                    batch_size, world_size
                )

                for aaa in range(iterations):
                    optimizer.zero_grad()
                    all_outputs, all_labels = get_all_outputs_and_labels(
                        inputs, labels, original_model, aaa
                    )
                    if use_ref:
                        all_ref_outputs, all_ref_labels = get_all_outputs_and_labels(
                            ref_inputs, ref_labels, original_model, aaa
                        )

                    indices_tuple = None
                    if efficient:
                        losses = []
                        for i in range(len(inputs[aaa])):
                            curr_emb, other_emb = create_efficient_batch(
                                all_outputs, all_ref_outputs, i, batch_size
                            )
                            curr_labels, other_labels = create_efficient_batch(
                                all_labels, all_ref_labels, i, batch_size
                            )
                            if original_miner_fn:
                                indices_tuple = distributed.get_indices_tuple(
                                    curr_labels,
                                    other_labels,
                                    embeddings=curr_emb,
                                    ref_emb=other_emb,
                                    miner=original_miner_fn,
                                )
                            else:
                                indices_tuple = distributed.get_indices_tuple(
                                    curr_labels, other_labels
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
                            indices_tuple = original_miner_fn(
                                all_outputs, all_labels, all_ref_outputs, all_ref_labels
                            )
                        if xbm:
                            enqueue_idx = (
                                global_enqueue_idx if use_xbm_enqueue_idx else None
                            )
                            loss = original_loss_fn(
                                all_outputs, all_labels, indices_tuple, enqueue_idx
                            )
                        else:
                            loss = original_loss_fn(
                                all_outputs,
                                all_labels,
                                indices_tuple,
                                all_ref_outputs,
                                all_ref_labels,
                            )

                    loss.backward()
                    optimizer.step()

                mp.spawn(
                    single_process_function,
                    args=(
                        world_size,
                        lr,
                        iterations,
                        model,
                        inputs,
                        labels,
                        ref_inputs,
                        ref_labels,
                        loss_fn,
                        miner_fn,
                        original_model,
                        efficient,
                        pass_labels_to_loss_fn,
                        use_xbm_enqueue_idx,
                        local_enqueue_idx,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    def test_distributed_tuple_loss(self):
        for xbm in [False, True]:
            for use_ref in [False, True]:
                for use_xbm_enqueue_idx in [False, True]:
                    if xbm and use_ref:
                        continue
                    self.loss_and_miner_tester(
                        ContrastiveLoss,
                        None,
                        False,
                        xbm,
                        use_ref,
                        use_xbm_enqueue_idx=use_xbm_enqueue_idx,
                    )

    def test_distributed_tuple_loss_and_miner(self):
        for xbm in [False, True]:
            for use_ref in [False, True]:
                for pass_labels_to_loss_fn in [False, True]:
                    if xbm and use_ref or xbm and not pass_labels_to_loss_fn:
                        continue
                    self.loss_and_miner_tester(
                        ContrastiveLoss,
                        PairMarginMiner,
                        False,
                        xbm,
                        use_ref,
                        miner_kwargs={"pos_margin": 0.5, "neg_margin": 0.5},
                        pass_labels_to_loss_fn=pass_labels_to_loss_fn,
                    )

    def test_distributed_tuple_loss_efficient(self):
        for use_ref in [False, True]:
            self.loss_and_miner_tester(ContrastiveLoss, None, True, False, use_ref)

    def test_distributed_tuple_loss_and_miner_efficient(self):
        for use_ref in [False, True]:
            for pass_labels_to_loss_fn in [False, True]:
                self.loss_and_miner_tester(
                    ContrastiveLoss,
                    PairMarginMiner,
                    True,
                    False,
                    use_ref,
                    miner_kwargs={"pos_margin": 0.5, "neg_margin": 0.5},
                    pass_labels_to_loss_fn=pass_labels_to_loss_fn,
                )

    def test_single_proc(self):
        setup(0, 1)
        _loss_fn = ContrastiveLoss()
        _miner_fn = PairMarginMiner()
        loss_fn = distributed.DistributedLossWrapper(loss=_loss_fn)
        miner_fn = distributed.DistributedMinerWrapper(miner=_miner_fn)

        emb = torch.randn(32, 128, device=TEST_DEVICE)
        labels = torch.randint(0, 3, size=(32,))
        pairs = miner_fn(emb, labels)
        loss = loss_fn(emb, labels, indices_tuple=pairs)
        cleanup()

        self.assertEqual(loss, _loss_fn(emb, indices_tuple=_miner_fn(emb, labels)))


if __name__ == "__main__":
    unittest.main()
