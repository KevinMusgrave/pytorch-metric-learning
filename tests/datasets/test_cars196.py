import unittest
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_metric_learning.datasets.cars196 import Cars196
import shutil
import os

class TestCars196(unittest.TestCase):
    
    CARS_196_ROOT = "test_cars196"
    ALREADY_EXISTS = False

    # In the rare case the user has an already existing directory, do not delete it
    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.CARS_196_ROOT):
            cls.ALREADY_EXISTS = True

    def test_Cars196(self):
        train_test_data = Cars196(root=TestCars196.CARS_196_ROOT, split="train+test", download=True)
        train_data = Cars196(root=TestCars196.CARS_196_ROOT, split="train", download=True)
        test_data = Cars196(root=TestCars196.CARS_196_ROOT, split="test", download=False)

        self.assertTrue(len(train_test_data) == 16185)
        self.assertTrue(len(train_data) == 8054)
        self.assertTrue(len(test_data) == 8131)

    def test_CARS_196_dataloader(self):
        test_data = Cars196(
            root=TestCars196.CARS_196_ROOT, 
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
            shutil.rmtree(cls.CARS_196_ROOT)