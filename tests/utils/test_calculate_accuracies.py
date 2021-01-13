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

    def test_get_lone_query_labels_custom(self):
        def fn1(x, y):
            return abs(x - y) < 2

        def fn2(x, y):
            return abs(x - y) > 99

        query_labels = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

        for comparison_fn in [fn1, fn2]:
            correct_unique_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

            if comparison_fn is fn1:
                correct_counts = np.array([3, 4, 3, 3, 3, 3, 3, 3, 3, 2, 1])
                correct_lone_query_labels = np.array([100])
                correct_not_lone_query_mask = np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                    ]
                )
            elif comparison_fn is fn2:
                correct_counts = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2])
                correct_lone_query_labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
                correct_not_lone_query_mask = np.array(
                    [
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                    ]
                )

            label_counts, num_k = accuracy_calculator.get_label_match_counts(
                query_labels,
                query_labels,
                comparison_fn,
            )
            unique_labels, counts = label_counts

            self.assertTrue(np.all(unique_labels == correct_unique_labels))
            self.assertTrue(np.all(counts == correct_counts))

            (
                lone_query_labels,
                not_lone_query_mask,
            ) = accuracy_calculator.get_lone_query_labels(
                query_labels, label_counts, True, comparison_fn
            )

            self.assertTrue(np.all(lone_query_labels == correct_lone_query_labels))
            self.assertTrue(np.all(not_lone_query_mask == correct_not_lone_query_mask))

    def test_get_lone_query_labels_multi_dim(self):
        def equality2D(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] == y[..., 1])

        def custom_label_comparison_fn(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

        query_labels = np.array(
            [
                (1, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (1, 2),
                (4, 5),
            ]
        )

        for comparison_fn in [equality2D, custom_label_comparison_fn]:
            label_counts, num_k = accuracy_calculator.get_label_match_counts(
                query_labels,
                query_labels,
                comparison_fn,
            )

            unique_labels, counts = label_counts
            correct_unique_labels = np.array([[0, 3], [1, 2], [1, 3], [4, 5]])
            if comparison_fn is equality2D:
                correct_counts = np.array([3, 1, 1, 1])
            else:
                correct_counts = np.array([0, 1, 1, 0])

            self.assertTrue(np.all(correct_counts == counts))
            self.assertTrue(np.all(correct_unique_labels == unique_labels))

            if comparison_fn is equality2D:
                correct = [
                    (
                        True,
                        np.array([[1, 2], [1, 3], [4, 5]]),
                        np.array([False, True, True, True, False, False]),
                    ),
                    (
                        False,
                        np.array([[]]),
                        np.array([True, True, True, True, True, True]),
                    ),
                ]
            else:
                correct = [
                    (
                        True,
                        np.array([[0, 3], [4, 5]]),
                        np.array([True, False, False, False, True, False]),
                    ),
                    (
                        False,
                        np.array([[0, 3], [4, 5]]),
                        np.array([True, False, False, False, True, False]),
                    ),
                ]

            for same_source, correct_lone, correct_mask in correct:
                (
                    lone_query_labels,
                    not_lone_query_mask,
                ) = accuracy_calculator.get_lone_query_labels(
                    query_labels, label_counts, same_source, comparison_fn
                )
                if correct_lone.size == 0:
                    self.assertTrue(lone_query_labels.size == 0)
                else:
                    self.assertTrue(np.all(lone_query_labels == correct_lone))

                self.assertTrue(np.all(not_lone_query_mask == correct_mask))

    def test_get_lone_query_labels(self):
        query_labels = np.array([0, 1, 2, 3, 4, 5, 6])
        reference_labels = np.array([0, 0, 0, 1, 2, 2, 3, 4, 5])
        label_counts, _ = accuracy_calculator.get_label_match_counts(
            query_labels,
            reference_labels,
            accuracy_calculator.EQUALITY,
        )

        for same_source, correct in [
            (True, np.array([1, 3, 4, 5, 6])),
            (False, np.array([6])),
        ]:
            lone_query_labels, _ = accuracy_calculator.get_lone_query_labels(
                query_labels,
                label_counts,
                same_source,
                accuracy_calculator.EQUALITY,
            )
            self.assertTrue(np.all(lone_query_labels == correct))


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

        self.assertRaises(
            NotImplementedError,
            lambda: accuracy_calculator.AccuracyCalculator(
                include=("NMI", "AMI"),
                avg_of_avgs=False,
                label_comparison_fn=label_comparison_fn,
            ),
        )

        AC_global_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=False,
            label_comparison_fn=label_comparison_fn,
        )

        AC_per_class_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=True,
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

        # SIMPLE CASE
        query = np.arange(2)[:, None].astype(np.float32)
        reference = np.arange(5)[:, None].astype(np.float32)
        query_labels = np.array(
            [
                (1, 3),
                (7, 3),
            ]
        )
        reference_labels = np.array(
            [
                (1, 3),
                (7, 4),
                (1, 4),
                (1, 5),
                (1, 6),
            ]
        )

        correct_precision_at_1 = 0.5
        correct_r_precision = (1.0 / 3 + 1) / 2
        correct_mapr = ((1.0 / 3) / 3 + 1) / 2

        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(acc["precision_at_1"] == correct_precision_at_1)
        self.assertTrue(acc["r_precision"] == correct_r_precision)
        self.assertTrue(acc["mean_average_precision_at_r"] == correct_mapr)

        acc = AC_per_class_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(acc["precision_at_1"] == correct_precision_at_1)
        self.assertTrue(acc["r_precision"] == correct_r_precision)
        self.assertTrue(acc["mean_average_precision_at_r"] == correct_mapr)

    def test_accuracy_calculator_float_custom_comparison_function(self):
        def label_comparison_fn(x, y):
            return np.abs(x - y) < 1

        self.assertRaises(
            NotImplementedError,
            lambda: accuracy_calculator.AccuracyCalculator(
                include=("NMI", "AMI"),
                avg_of_avgs=False,
                label_comparison_fn=label_comparison_fn,
            ),
        )

        AC_global_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=False,
            label_comparison_fn=label_comparison_fn,
        )

        query = np.array([0, 3])[:, None].astype(np.float32)
        reference = np.arange(4)[:, None].astype(np.float32)
        query_labels = np.array(
            [
                0.01,
                0.02,
            ]
        )
        reference_labels = np.array(
            [
                10.0,
                0.03,
                0.04,
                0.05,
            ]
        )

        correct = {
            "precision_at_1": (0 + 1) / 2,
            "r_precision": (2 / 3 + 3 / 3) / 2,
            "mean_average_precision_at_r": ((0 + 1 / 2 + 2 / 3) / 3 + 1) / 2,
        }
        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        for k in correct:
            self.assertTrue(acc[k] == correct[k])

        correct = {
            "precision_at_1": (1 + 1) / 2,
            "r_precision": (3 / 3 + 2 / 3) / 2,
            "mean_average_precision_at_r": (1 + (1 / 1 + 2 / 2 + 0) / 3) / 2,
        }
        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, True
        )
        for k in correct:
            self.assertTrue(acc[k] == correct[k])
