import logging
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import WITH_COLLECT_STATS


def process_label_helper(
    cls,
    dataset_labels,
    batch_labels,
    correct_mapped_batch_labels,
    correct_unmapped_batch_labels,
    hierarchy_level,
    input_is_string=False,
):
    for set_min_label_to_zero in [False, True]:
        if input_is_string and not set_min_label_to_zero:
            continue
        label_mapper = c_f.LabelMapper(set_min_label_to_zero, dataset_labels)
        x = c_f.process_label(batch_labels, hierarchy_level, label_mapper.map)
        if set_min_label_to_zero:
            cls.assertTrue(np.array_equal(x, correct_mapped_batch_labels))
        else:
            cls.assertTrue(np.array_equal(x, correct_unmapped_batch_labels))


def process_label_helper2d(
    cls, dataset_labels, batch_labels, mapped_batch_labels, input_is_string=False
):
    for h in [0, 1]:
        process_label_helper(
            cls,
            dataset_labels,
            batch_labels,
            mapped_batch_labels[h],
            batch_labels[:, h],
            h,
            input_is_string=input_is_string,
        )


class TestCommonFunctions(unittest.TestCase):
    def test_torch_standard_scaler(self):
        torch.manual_seed(56987)
        embeddings = torch.randn(1024, 512)
        scaled = c_f.torch_standard_scaler(embeddings)
        true_scaled = StandardScaler().fit_transform(embeddings.cpu().numpy())
        true_scaled = torch.from_numpy(true_scaled)
        self.assertTrue(torch.all(torch.isclose(scaled, true_scaled, rtol=1e-2)))

    def test_collect_stats_flag(self):
        self.assertTrue(c_f.COLLECT_STATS == WITH_COLLECT_STATS)
        loss_fn = TripletMarginLoss()
        self.assertTrue(loss_fn.collect_stats == WITH_COLLECT_STATS)
        self.assertTrue(loss_fn.distance.collect_stats == WITH_COLLECT_STATS)
        self.assertTrue(loss_fn.reducer.collect_stats == WITH_COLLECT_STATS)

    def test_check_shapes(self):
        embeddings = torch.randn(32, 512, 3)
        labels = torch.randn(32)
        loss_fn = TripletMarginLoss()

        # embeddings is 3-dimensional
        self.assertRaises(ValueError, lambda: loss_fn(embeddings, labels))

        # embeddings does not match labels
        embeddings = torch.randn(33, 512)
        self.assertRaises(ValueError, lambda: loss_fn(embeddings, labels))

        # labels is 2D
        embeddings = torch.randn(32, 512)
        labels = labels.unsqueeze(1)
        self.assertRaises(ValueError, lambda: loss_fn(embeddings, labels))

        # correct shapes
        labels = labels.squeeze(1)
        self.assertTrue(torch.is_tensor(loss_fn(embeddings, labels)))

    def test_logger(self):
        logging.basicConfig()
        for name in [None, "some_random_name"]:
            if name is not None:
                c_f.set_logger_name(name)
                self.assertTrue(c_f.LOGGER_NAME == name)
            for level in range(0, 60, 10):
                logging.getLogger(c_f.LOGGER_NAME).setLevel(level)
                self.assertTrue(c_f.LOGGER.level == level)
                self.assertTrue(logging.getLogger().level == logging.WARNING)

    def test_process_label(self):
        # 1D number labels
        dataset_labels = np.array([1, 1, 10, 13, 13, 13, 15])
        batch_labels = np.array([1, 10, 1, 13, 10, 15, 15, 15])
        mapped_batch_labels = np.array([0, 1, 0, 2, 1, 3, 3, 3])
        process_label_helper(
            self, dataset_labels, batch_labels, mapped_batch_labels, batch_labels, 0
        )

        # 1D string labels
        # These work only with set_min_label_to_zero=True
        # The labels will be sorted alphabetically. So in the following, "bear" becomes 0
        dataset_labels = np.array(["dog", "dog", "cat", "bear", "bear", "bear", "lion"])
        batch_labels = np.array(
            ["dog", "cat", "dog", "bear", "cat", "lion", "lion", "lion"]
        )
        mapped_batch_labels = np.array([2, 1, 2, 0, 1, 3, 3, 3])
        process_label_helper(
            self,
            dataset_labels,
            batch_labels,
            mapped_batch_labels,
            batch_labels,
            0,
            input_is_string=True,
        )

        # 2D number labels
        dataset_labels = np.array(
            [[1, 1, 10, 13, 13, 13, 15], [10, 20, 30, 40, 50, 60, 70]]
        ).transpose()
        batch_labels = np.array(
            [[1, 10, 1, 13, 10, 15, 15, 15], [30, 70, 40, 10, 70, 60, 50, 40]]
        ).transpose()
        mapped_batch_labels0 = np.array([0, 1, 0, 2, 1, 3, 3, 3])
        mapped_batch_labels1 = np.array([2, 6, 3, 0, 6, 5, 4, 3])
        process_label_helper2d(
            self,
            dataset_labels,
            batch_labels,
            [mapped_batch_labels0, mapped_batch_labels1],
        )

        # 2D string labels
        dataset_labels = np.array(
            [
                ["dog", "dog", "cat", "bear", "bear", "bear", "lion"],
                ["A.1", "A.2", "A.3", "A.4", "A.5", "A.6", "A.7"],
            ]
        ).transpose()
        batch_labels = np.array(
            [
                ["dog", "cat", "dog", "bear", "cat", "lion", "lion", "lion"],
                ["A.3", "A.7", "A.4", "A.1", "A.7", "A.6", "A.5", "A.4"],
            ]
        ).transpose()
        mapped_batch_labels0 = np.array([2, 1, 2, 0, 1, 3, 3, 3])
        mapped_batch_labels1 = np.array([2, 6, 3, 0, 6, 5, 4, 3])
        process_label_helper2d(
            self,
            dataset_labels,
            batch_labels,
            [mapped_batch_labels0, mapped_batch_labels1],
            input_is_string=True,
        )


if __name__ == "__main__":
    unittest.main()
