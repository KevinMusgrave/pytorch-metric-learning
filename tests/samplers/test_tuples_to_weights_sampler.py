import unittest
import torch
from pytorch_metric_learning.samplers import TuplesToWeightsSampler
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.miners import MultiSimilarityMiner
from torchvision import models, datasets, transforms
import shutil
from .. import TEST_DEVICE
import os


class TestTuplesToWeightsSampler(unittest.TestCase):
    def test_tuplestoweights_sampler(self):
        model = models.resnet18(pretrained=True)
        model.fc = c_f.Identity()
        model = torch.nn.DataParallel(model)
        model.to(TEST_DEVICE)

        miner = MultiSimilarityMiner(epsilon=-0.2)

        eval_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        temporary_folder = "cifar100_temp_for_pytorch_metric_learning_test"

        assert not os.path.isdir(temporary_folder)

        dataset = datasets.CIFAR100(
            temporary_folder, train=True, download=True, transform=eval_transform
        )
        subset_size = 1000
        sampler = TuplesToWeightsSampler(model, miner, dataset, subset_size=subset_size)
        iterable_as_list = list(iter(sampler))
        self.assertTrue(len(iterable_as_list) == subset_size)
        unique_idx = torch.unique(torch.tensor(iterable_as_list))
        self.assertTrue(torch.all(sampler.weights[unique_idx] != 0))

        shutil.rmtree(temporary_folder)


if __name__ == "__main__":
    unittest.main()
