import unittest

import torch

from pytorch_metric_learning.testers import GlobalEmbeddingSpaceTester
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils import common_functions as c_f


class TestGlobalEmbeddingSpaceTester(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        embedding_angles = [0, 10, 20, 30, 50, 60, 70, 80]
        embeddings1 = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles])
        labels1 = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])

        embedding_angles = [1, 11, 21, 31, 51, 59, 71, 81]
        embeddings2 = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles])
        labels2 = torch.LongTensor([1, 1, 1, 1, 1, 0, 0, 0])

        self.dataset_dict = {
            "train": c_f.EmbeddingDataset(embeddings1, labels1),
            "val": c_f.EmbeddingDataset(embeddings2, labels2),
        }

    def test_global_embedding_space_tester(self):
        model = c_f.Identity()
        AC = accuracy_calculator.AccuracyCalculator(include=("precision_at_1",))

        correct = [
            (None, {"train": 1, "val": 6.0 / 8}),
            (
                [("train", ["train", "val"]), ("val", ["train", "val"])],
                {"train": 1.0 / 8, "val": 1.0 / 8},
            ),
            ([("train", ["train"]), ("val", ["train"])], {"train": 1, "val": 1.0 / 8}),
        ]

        for splits_to_eval, correct_vals in correct:
            tester = GlobalEmbeddingSpaceTester(accuracy_calculator=AC)
            tester.test(self.dataset_dict, 0, model, splits_to_eval=splits_to_eval)
            self.assertTrue(
                tester.all_accuracies["train"]["precision_at_1_level0"]
                == correct_vals["train"]
            )
            self.assertTrue(
                tester.all_accuracies["val"]["precision_at_1_level0"]
                == correct_vals["val"]
            )

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
