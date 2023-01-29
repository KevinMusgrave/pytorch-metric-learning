import unittest

import numpy as np
import torch

from pytorch_metric_learning.testers import WithSameParentLabelTester
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils import common_functions as c_f

from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestWithSameParentLabelTester(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        embedding_angles = [0, 9, 21, 29, 31, 39, 51, 59]
        embeddings1 = torch.tensor([angle_to_coord(a) for a in embedding_angles])
        parent_labels1 = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
        child_labels1 = torch.LongTensor([2, 2, 3, 3, 4, 4, 5, 5])
        labels1 = torch.stack([child_labels1, parent_labels1], axis=1)

        embedding_angles = [2, 11, 23, 32, 33, 41, 53, 89, 90]
        embeddings2 = torch.tensor([angle_to_coord(a) for a in embedding_angles])
        parent_labels2 = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
        child_labels2 = torch.LongTensor([2, 2, 4, 3, 4, 4, 4, 5, 5])
        labels2 = torch.stack([child_labels2, parent_labels2], axis=1)

        self.dataset_dict = {
            "train": c_f.EmbeddingDataset(embeddings1, labels1),
            "val": c_f.EmbeddingDataset(embeddings2, labels2),
        }

    def test_with_same_parent_label_tester(self):
        model = torch.nn.Identity()
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",))

        correct = [
            (None, {"train": 1, "val": (1 + 0.5) / 2}),
            (
                [("train", ["train", "val"]), ("val", ["train", "val"])],
                {
                    "train": (0.75 + 0.5) / 2,
                    "val": (0.4 + 0.75) / 2,
                },
            ),
            (
                [("train", ["train"]), ("val", ["train"])],
                {"train": 1, "val": (1 + 0.75) / 2},
            ),
        ]

        for splits_to_eval, correct_vals in correct:
            tester = WithSameParentLabelTester(accuracy_calculator=AC)
            all_accuracies = tester.test(
                self.dataset_dict, 0, model, splits_to_eval=splits_to_eval
            )
            self.assertTrue(
                np.isclose(
                    all_accuracies["train"]["precision_at_1_level0"],
                    correct_vals["train"],
                )
            )
            self.assertTrue(
                np.isclose(
                    all_accuracies["val"]["precision_at_1_level0"], correct_vals["val"]
                )
            )

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
