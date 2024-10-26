import unittest
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_metric_learning.datasets.sop import StanfordOnlineProducts
import shutil
import os

class TestStanfordOnlineProducts(unittest.TestCase):
    
    SOP_ROOT = "test_sop"
    ALREADY_EXISTS = False

    # In the rare case the user has an already existing directory, do not delete it
    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.SOP_ROOT):
            cls.ALREADY_EXISTS = True

    def test_SOP(self):
        train_test_data = StanfordOnlineProducts(
            root=TestStanfordOnlineProducts.SOP_ROOT, split="train+test", download=True)
        train_data = StanfordOnlineProducts(
            root=TestStanfordOnlineProducts.SOP_ROOT, split="train", download=True)
        test_data = StanfordOnlineProducts(
            root=TestStanfordOnlineProducts.SOP_ROOT, split="test", download=False)

        self.assertTrue(len(train_test_data) == 120053)
        self.assertTrue(len(train_data) == 59551)
        self.assertTrue(len(test_data) == 60502)

    def test_SOP_dataloader(self):
        test_data = StanfordOnlineProducts(
            root=TestStanfordOnlineProducts.SOP_ROOT, 
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
            shutil.rmtree(cls.SOP_ROOT)