import os
import unittest
import uuid

import faiss
import torch
import torchvision
from torchvision import datasets, transforms

from pytorch_metric_learning.utils.inference import InferenceModel

from .. import TEST_DEVICE


class TextModel(torch.nn.Module):
    def forward(self, list_of_text):
        return torch.randn(len(list_of_text), 32)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [str(uuid.uuid4()) for _ in range(500)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trunk = torchvision.models.resnet18(pretrained=True)
        trunk.fc = torch.nn.Identity()
        trunk = trunk.to(TEST_DEVICE)

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

        cls.train_vectors = [cls.dataset[i][0] for i in range(len(cls.dataset))]

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_untrained_indexer(self):
        inference_model = InferenceModel(trunk=self.model, data_device=TEST_DEVICE)
        with self.assertRaises(ValueError):
            inference_model.get_nearest_neighbors(self.dataset[0][0].unsqueeze(0), k=10)

    def test_get_nearest_neighbors(self):
        test_filename = "test_inference.index"
        for indexer_input in [self.train_vectors, self.dataset]:
            for load_from_file in [False, True]:
                inference_model = InferenceModel(
                    trunk=self.model, data_device=TEST_DEVICE
                )
                if load_from_file:
                    inference_model.load_knn_func(test_filename)
                else:
                    inference_model.train_knn(indexer_input)
                    inference_model.save_knn_func(test_filename)

                self.helper_assertions(inference_model)

        os.remove(test_filename)

    def test_add_to_indexer(self):
        inference_model = InferenceModel(trunk=self.model, data_device=TEST_DEVICE)
        inference_model.knn_func.index = faiss.IndexFlatL2(512)
        inference_model.add_to_knn(self.dataset)
        self.helper_assertions(inference_model)

    def test_list_of_text(self):
        model = TextModel()
        dataset = TextDataset()
        inference_model = InferenceModel(trunk=model, data_device=TEST_DEVICE)
        inference_model.train_knn(dataset)
        inference_model.add_to_knn([["test1", "test2"], ["test3", "test4"]])
        inference_model.get_nearest_neighbors(["dog", "cat"], k=10)

    def helper_assertions(self, inference_model):
        distances, indices = inference_model.get_nearest_neighbors(
            [self.train_vectors[0]], k=10
        )
        # The closest image is the query itself
        self.assertTrue(indices[0][0] == 0)
        self.assertTrue(len(indices) == 1)
        self.assertTrue(len(distances) == 1)
        self.assertTrue(len(indices[0]) == 10)
        self.assertTrue(len(distances[0]) == 10)
