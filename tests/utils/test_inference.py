import unittest
import torch
import torchvision
from torchvision import datasets, transforms
from pytorch_metric_learning.utils.inference import InferenceModel
from pytorch_metric_learning.utils import common_functions
from .. import TEST_DEVICE


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trunk = torchvision.models.resnet18(pretrained=True)
        cls.emb_dim = trunk.fc.in_features
        trunk.fc = common_functions.Identity()
        trunk = torch.nn.DataParallel(trunk.to(TEST_DEVICE))

        cls.model = trunk

        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        cls.dataset = datasets.FakeData(
            size=200, image_size=(3, 64, 64), transform=transform
        )

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_untrained_indexer(self):
        inference_model = InferenceModel(trunk=self.model)
        with self.assertRaises(RuntimeError):
            inference_model.get_nearest_neighbors(self.dataset[0][0], k=10)

    def test_get_nearest_neighbors(self):
        inference_model = InferenceModel(trunk=self.model)

        train_vectors = [self.dataset[i][0] for i in range(len(self.dataset))]
        inference_model.train_indexer(train_vectors, self.emb_dim)

        self.assertTrue(inference_model.indexer.index.is_trained)

        indices, distances = inference_model.get_nearest_neighbors(
            [train_vectors[0]], k=10
        )
        # The closest image is the query itself
        self.assertTrue(indices[0][0] == 0)
        self.assertTrue(len(indices) == 1)
        self.assertTrue(len(distances) == 1)
        self.assertTrue(len(indices[0]) == 10)
        self.assertTrue(len(distances[0]) == 10)

        self.assertTrue((indices != -1).any())
