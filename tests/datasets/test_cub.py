import unittest
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_metric_learning.datasets.cub import CUB
import shutil
import os

class TestCUB(unittest.TestCase):
    
    CUB_ROOT = "test_cub"
    ALREADY_EXISTS = False

    # In the rare case the user has an already existing directory, do not delete it
    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.CUB_ROOT):
            cls.ALREADY_EXISTS = True

    def test_CUB(self):
        train_test_data = CUB(root=TestCUB.CUB_ROOT, split="train+test", download=True)
        train_data = CUB(root=TestCUB.CUB_ROOT, split="train", download=True)
        test_data = CUB(root=TestCUB.CUB_ROOT, split="test", download=False)

        self.assertTrue(len(train_test_data) == 11788)
        self.assertTrue(len(train_data) == 5864)
        self.assertTrue(len(test_data) == 5924)

    def test_CUB_dataloader(self):
        test_data = CUB(
            root=TestCUB.CUB_ROOT, 
            transform=transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.PILToTensor()
            ]),
            split="test", 
            download=True
        )
        loader = DataLoader(test_data, batch_size=8)
        inputs, labels = next(iter(loader))
        self.assertTupleEqual(tuple(inputs.shape), (8, 3, 224, 224))
        self.assertTupleEqual(tuple(labels.shape), (8,))

    @classmethod
    def tearDownClass(cls):
        if not cls.ALREADY_EXISTS:
            shutil.rmtree(cls.CUB_ROOT)