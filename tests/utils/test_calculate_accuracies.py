import unittest

import numpy as np

from pytorch_metric_learning.utils import accuracy_calculator


class TestCalculateAccuracies(unittest.TestCase):
    def test_accuracy_calculator(self):
        query_labels = np.array([1, 1, 2, 3, 4])

        knn_labels1 = np.array(
            [
                [0, 1, 1, 2, 2],
                [1, 0, 1, 1, 3],
                [4, 4, 4, 4, 2],
                [3, 1, 3, 1, 3],
                [0, 0, 4, 2, 2],
            ]
        )
        label_counts1 = ([1, 2, 3, 4], [3, 5, 4, 5])

        knn_labels2 = knn_labels1 + 5
        label_counts2 = ([6, 7, 8, 9], [3, 5, 4, 5])

        for avg_of_avgs in [False, True]:
            for i, (knn_labels, label_counts) in enumerate(
                [(knn_labels1, label_counts1), (knn_labels2, label_counts2)]
            ):

                AC = accuracy_calculator.AccuracyCalculator(
                    exclude=("NMI", "AMI"), avg_of_avgs=avg_of_avgs
                )
                kwargs = {
                    "query_labels": query_labels,
                    "label_counts": label_counts,
                    "knn_labels": knn_labels,
                    "not_lone_query_mask": np.ones(5).astype(np.bool)
                    if i == 0
                    else np.zeros(5).astype(np.bool),
                }

                function_dict = AC.get_function_dict()

                for ecfss in [False, True]:
                    if ecfss:
                        kwargs["knn_labels"] = kwargs["knn_labels"][:, 1:]
                    kwargs["embeddings_come_from_same_source"] = ecfss
                    acc = AC._get_accuracy(function_dict, **kwargs)
                    if i == 1:
                        self.assertTrue(acc["precision_at_1"] == 0)
                        self.assertTrue(acc["r_precision"] == 0)
                        self.assertTrue(acc["mean_average_precision_at_r"] == 0)
                        self.assertTrue(acc["mean_average_precision"] == 0)
                    else:
                        self.assertTrue(
                            acc["precision_at_1"]
                            == self.correct_precision_at_1(ecfss, avg_of_avgs)
                        )
                        self.assertTrue(
                            acc["r_precision"]
                            == self.correct_r_precision(ecfss, avg_of_avgs)
                        )
                        self.assertTrue(
                            acc["mean_average_precision_at_r"]
                            == self.correct_mean_average_precision_at_r(
                                ecfss, avg_of_avgs
                            )
                        )
                        self.assertTrue(
                            acc["mean_average_precision"]
                            == self.correct_mean_average_precision(ecfss, avg_of_avgs)
                        )

    def correct_precision_at_1(self, embeddings_come_from_same_source, avg_of_avgs):
        if not embeddings_come_from_same_source:
            if not avg_of_avgs:
                return 0.4
            else:
                return (0.5 + 0 + 1 + 0) / 4
        else:
            if not avg_of_avgs:
                return 1.0 / 5
            else:
                return (0.5 + 0 + 0 + 0) / 4

    def correct_r_precision(self, embeddings_come_from_same_source, avg_of_avgs):
        if not embeddings_come_from_same_source:
            acc0 = 2.0 / 3
            acc1 = 2.0 / 3
            acc2 = 1.0 / 5
            acc3 = 2.0 / 4
            acc4 = 1.0 / 5
        else:
            acc0 = 1.0 / 1
            acc1 = 1.0 / 2
            acc2 = 1.0 / 4
            acc3 = 1.0 / 3
            acc4 = 1.0 / 4
        if not avg_of_avgs:
            return np.mean([acc0, acc1, acc2, acc3, acc4])
        else:
            return np.mean([(acc0 + acc1) / 2, acc2, acc3, acc4])

    def correct_mean_average_precision_at_r(
        self, embeddings_come_from_same_source, avg_of_avgs
    ):
        if not embeddings_come_from_same_source:
            acc0 = (1.0 / 2 + 2.0 / 3) / 3
            acc1 = (1 + 2.0 / 3) / 3
            acc2 = (1.0 / 5) / 5
            acc3 = (1 + 2.0 / 3) / 4
            acc4 = (1.0 / 3) / 5
        else:
            acc0 = 1
            acc1 = (1.0 / 2) / 2
            acc2 = (1.0 / 4) / 4
            acc3 = (1.0 / 2) / 3
            acc4 = (1.0 / 2) / 4
        if not avg_of_avgs:
            return np.mean([acc0, acc1, acc2, acc3, acc4])
        else:
            return np.mean([(acc0 + acc1) / 2, acc2, acc3, acc4])

    def correct_mean_average_precision(
        self, embeddings_come_from_same_source, avg_of_avgs
    ):
        if not embeddings_come_from_same_source:
            acc0 = (1.0 / 2 + 2.0 / 3) / 2
            acc1 = (1 + 2.0 / 3 + 3.0 / 4) / 3
            acc2 = (1.0 / 5) / 1
            acc3 = (1 + 2.0 / 3 + 3.0 / 5) / 3
            acc4 = (1.0 / 3) / 1
        else:
            acc0 = 1
            acc1 = (1.0 / 2 + 2.0 / 3) / 2
            acc2 = 1.0 / 4
            acc3 = (1.0 / 2 + 2.0 / 4) / 2
            acc4 = 1.0 / 2
        if not avg_of_avgs:
            return np.mean([acc0, acc1, acc2, acc3, acc4])
        else:
            return np.mean([(acc0 + acc1) / 2, acc2, acc3, acc4])

    def test_get_label_match_counts(self):
        (unique_labels, counts), num_k = accuracy_calculator.get_label_match_counts(
            [0, 1, 3, 2, 3, 1, 3, 3, 4, 6, 5, 10, 4, 4, 4, 4, 6, 6, 5],
            accuracy_calculator.EQUALITY,
        )
        self.assertTrue(sorted(unique_labels) == sorted([0, 1, 2, 3, 4, 5, 6, 10]))
        self.assertTrue(sorted(counts) == sorted([1, 2, 1, 4, 5, 2, 3, 1]))
        self.assertTrue(num_k == 5)

    def test_get_lone_query_labels(self):
        query_labels = np.array([0, 1, 2, 3, 4, 5, 6])
        reference_labels = np.array([0, 0, 0, 1, 2, 2, 3, 4, 5, 6])
        reference_label_counts, _ = accuracy_calculator.get_label_match_counts(
            reference_labels,
            accuracy_calculator.EQUALITY,
        )

        lone_query_labels, _ = accuracy_calculator.get_lone_query_labels(
            query_labels,
            reference_label_counts,
            True,
            accuracy_calculator.EQUALITY,
        )
        self.assertTrue(
            np.all(np.unique(lone_query_labels) == np.array([1, 3, 4, 5, 6]))
        )

        query_labels = np.array([0, 1, 2, 3, 4])
        reference_labels = np.array([0, 0, 0, 1, 2, 2, 4, 5, 6])

        lone_query_labels, _ = accuracy_calculator.get_lone_query_labels(
            query_labels,
            reference_label_counts,
            False,
            accuracy_calculator.EQUALITY,
        )
        self.assertTrue(np.all(np.unique(lone_query_labels) == np.array([3])))


class TestCalculateAccuraciesAndFaiss(unittest.TestCase):
    def test_accuracy_calculator_and_faiss(self):
        AC = accuracy_calculator.AccuracyCalculator(exclude=("NMI", "AMI"))

        query = np.arange(10)[:, None].astype(np.float32)
        reference = np.arange(10)[:, None].astype(np.float32)
        query_labels = np.arange(10).astype(np.int)
        reference_labels = np.arange(10).astype(np.int)
        acc = AC.get_accuracy(query, reference, query_labels, reference_labels, False)
        self.assertTrue(acc["precision_at_1"] == 1)
        self.assertTrue(acc["r_precision"] == 1)
        self.assertTrue(acc["mean_average_precision_at_r"] == 1)

        reference = (np.arange(20) / 2.0)[:, None].astype(np.float32)
        reference_labels = np.zeros(20).astype(np.int)
        reference_labels[::2] = query_labels
        reference_labels[1::2] = np.ones(10).astype(np.int)
        acc = AC.get_accuracy(query, reference, query_labels, reference_labels, True)
        self.assertTrue(acc["precision_at_1"] == 1)
        self.assertTrue(acc["r_precision"] == 0.5)
        self.assertTrue(
            acc["mean_average_precision_at_r"]
            == (1 + 2.0 / 2 + 3.0 / 5 + 4.0 / 7 + 5.0 / 9) / 10
        )

    def test_accuracy_calculator_and_faiss_avg_of_avgs(self):
        AC_global_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"), avg_of_avgs=False
        )
        AC_per_class_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"), avg_of_avgs=True
        )
        query = np.arange(10)[:, None].astype(np.float32)
        reference = np.arange(10)[:, None].astype(np.float32)
        query[-1] = 100
        reference[0] = -100
        query_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        reference_labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(acc["precision_at_1"] == 0.9)
        self.assertTrue(acc["r_precision"] == 0.9)
        self.assertTrue(acc["mean_average_precision_at_r"] == 0.9)

        acc = AC_per_class_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(acc["precision_at_1"] == 0.5)
        self.assertTrue(acc["r_precision"] == 0.5)
        self.assertTrue(acc["mean_average_precision_at_r"] == 0.5)

    def test_accuracy_calculator_custom_comparison_function(self):
        def label_comparison_fn(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

        AC = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=False,
            label_comparison_fn=label_comparison_fn,
        )
        query = np.arange(10)[:, None].astype(np.float32)
        reference = np.arange(10)[:, None].astype(np.float32)
        query[-1] = 100
        reference[0] = -100
        query_labels = np.array(
            [
                (0, 2),
                (0, 2),
                (0, 2),
                (0, 2),
                (0, 2),
                (0, 2),
                (0, 2),
                (0, 2),
                (0, 2),
                (1, 2),
            ]
        )
        reference_labels = np.array(
            [
                (1, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (0, 3),
            ]
        )
        acc = AC.get_accuracy(query, reference, query_labels, reference_labels, False)
        self.assertTrue(acc["precision_at_1"] == 0.9)
        self.assertTrue(acc["r_precision"] == 0.9)
        self.assertTrue(acc["mean_average_precision_at_r"] == 0.9)