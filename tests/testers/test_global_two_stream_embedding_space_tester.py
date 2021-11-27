import unittest

import numpy as np
import torch

from pytorch_metric_learning.testers import GlobalTwoStreamEmbeddingSpaceTester
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils import common_functions as c_f


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, anchors, positives, labels):
        self.anchors = anchors
        self.positives = positives
        self.labels = labels

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.labels[idx]


class TestGlobalTwoStreamEmbeddingSpaceTester(unittest.TestCase):
    def test_global_two_stream_embedding_space_tester(self):
        embedding_angles = [0, 10, 20, 30, 50, 60, 70, 80]
        embeddings1 = torch.tensor([angle_to_coord(a) for a in embedding_angles])
        embedding_angles = [81, 71, 61, 31, 51, 21, 11, 1]
        embeddings2 = torch.tensor([angle_to_coord(a) for a in embedding_angles])
        labels = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataset_dict = {
            "train": FakeDataset(embeddings1, embeddings2, labels),
        }

        model = c_f.Identity()
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",))

        tester = GlobalTwoStreamEmbeddingSpaceTester(
            accuracy_calculator=AC, dataloader_num_workers=0
        )
        all_accuracies = tester.test(dataset_dict, 0, model)

        self.assertTrue(
            np.isclose(all_accuracies["train"]["precision_at_1_level0"], 0.25)
        )

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
