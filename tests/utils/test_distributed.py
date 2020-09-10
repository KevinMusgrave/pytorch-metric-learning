import unittest 
import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from pytorch_metric_learning.utils import distributed
from pytorch_metric_learning import losses, miners
from torch.nn.parallel import DistributedDataParallel as DDP

# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
def parameters_are_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        num_elements = float(torch.numel(p2.data))
        if torch.sum(torch.isclose(p1.data, p2.data, rtol=1e-2)) < (num_elements*0.9):
            return False
    return True

### from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html ###
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

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
def single_process_function(rank, 
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
                            device):
    setup(rank, world_size)

    ddp_mp_model = DDP(model, device_ids=[rank])

    loss_fn = distributed.DistributedLossWrapper(loss=loss_fn)
    miner_fn = distributed.DistributedMinerWrapper(miner=miner_fn)

    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=lr)
    optimizer.zero_grad()
    outputs = ddp_mp_model(inputs[rank].to(device))
    indices_tuple = miner_fn(outputs, labels[rank])

    loss = loss_fn(outputs, labels[rank], indices_tuple)

    pos_loss_size = loss_fn.loss.reducer.reducers["pos_loss"].losses_size
    neg_loss_size = loss_fn.loss.reducer.reducers["neg_loss"].losses_size

    correct_pos_loss_size = original_loss_fn.reducer.reducers["pos_loss"].losses_size
    correct_neg_loss_size = original_loss_fn.reducer.reducers["neg_loss"].losses_size

    assert torch.isclose(loss, correct_loss)
    assert pos_loss_size == correct_pos_loss_size
    assert neg_loss_size == correct_neg_loss_size
    assert miner_fn.miner.num_pos_pairs == original_miner_fn.num_pos_pairs
    assert miner_fn.miner.num_neg_pairs == original_miner_fn.num_neg_pairs
    for i in range(len(correct_indices_tuple)):
        assert torch.all(indices_tuple[i] == correct_indices_tuple[i])

    loss.backward()

    assert not parameters_are_equal(original_model, ddp_mp_model.module)
    dist.barrier()
    optimizer.step()
    dist.barrier()
    assert parameters_are_equal(original_model, ddp_mp_model.module)

    cleanup()


class TestDistributedLossWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_distributed_loss(self):
        for world_size in range(1,5):
            batch_size = 20
            lr = 1
            inputs = [torch.randn(batch_size, 10) for _ in range(world_size)]
            labels = [torch.randint(low=0, high=2, size=(batch_size,)) for _ in range(world_size)]
            original_model = ToyMpModel().to(self.device)
            model = ToyMpModel().to(self.device)
            model.load_state_dict(original_model.state_dict())

            optimizer = optim.SGD(original_model.parameters(), lr=lr)
            optimizer.zero_grad()
            all_inputs = torch.cat(inputs, dim=0).to(self.device)
            all_labels = torch.cat(labels, dim=0).to(self.device)
            all_outputs = original_model(all_inputs)
            original_loss_fn = losses.ContrastiveLoss()
            original_miner_fn = miners.MultiSimilarityMiner()
            correct_indices_tuple = original_miner_fn(all_outputs, all_labels)
            correct_loss = original_loss_fn(all_outputs, all_labels, correct_indices_tuple)
            correct_loss.backward()
            optimizer.step()

            # need to make separate copy to do test properly
            loss_fn = losses.ContrastiveLoss()
            miner_fn = miners.MultiSimilarityMiner()

            mp.spawn(single_process_function,
                    args=(world_size,
                        lr, 
                        model, 
                        inputs, 
                        labels, 
                        loss_fn, 
                        miner_fn,
                        original_model, 
                        original_loss_fn,
                        original_miner_fn, 
                        correct_loss.detach(), 
                        correct_indices_tuple,
                        self.device),
                    nprocs=world_size,
                    join=True)


if __name__ == "__main__":
    unittest.main()