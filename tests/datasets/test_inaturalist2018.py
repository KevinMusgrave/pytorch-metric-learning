import unittest
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_metric_learning.datasets.inaturalist2018 import INaturalist2018
import shutil
import os

class TestINaturalist2018(unittest.TestCase):
    
    INATURALIST2018_ROOT = "data"
    ALREADY_EXISTS = False

    # In the rare case the user has an already existing directory, do not delete it
    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.INATURALIST2018_ROOT):
            cls.ALREADY_EXISTS = True

    def test_INaturalist2018(self):
        train_test_data = INaturalist2018(
            root=TestINaturalist2018.INATURALIST2018_ROOT, split="train+test", download=True
        )
        train_data = INaturalist2018(
            root=TestINaturalist2018.INATURALIST2018_ROOT, split="train", download=True
        )
        test_data = INaturalist2018(
            root=TestINaturalist2018.INATURALIST2018_ROOT, split="test", download=False
        )

        self.assertTrue(len(train_test_data) == 461939)
        self.assertTrue(len(train_data) == 325846)
        self.assertTrue(len(test_data) == 136093)

    def test_INaturalist2018_dataloader(self):
        test_data = INaturalist2018(
            root=TestINaturalist2018.INATURALIST2018_ROOT, 
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
            shutil.rmtree(cls.INATURALIST2018_ROOT)