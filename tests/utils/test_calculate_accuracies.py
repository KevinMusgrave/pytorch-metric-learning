import itertools
import unittest

import numpy as np
import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils.inference import CustomKNN, FaissKNN

from .. import TEST_DEVICE


def isclose(x, y):
    rtol = 0
    if TEST_DEVICE == torch.device("cpu"):
        atol = 1e-15
    else:
        atol = 1e-7
    return np.isclose(x, y, atol=atol, rtol=rtol)


class TestCalculateAccuracies(unittest.TestCase):
    def test_accuracy_calculator(self):
        query_labels = torch.tensor([1, 1, 2, 3, 4], device=TEST_DEVICE)

        knn_labels1 = torch.tensor(
            [
                [0, 1, 1, 2, 2],
                [1, 0, 1, 1, 3],
                [4, 4, 4, 4, 2],
                [3, 1, 3, 1, 3],
                [0, 0, 4, 2, 2],
            ],
            device=TEST_DEVICE,
        )
        label_counts1 = ([1, 2, 3, 4], [3, 5, 4, 5])

        knn_labels2 = knn_labels1 + 5
        label_counts2 = ([6, 7, 8, 9], [3, 5, 4, 5])

        for avg_of_avgs in [False, True]:
            for i, (knn_labels, label_counts) in enumerate(
                [(knn_labels1, label_counts1), (knn_labels2, label_counts2)]
            ):

                AC = accuracy_calculator.AccuracyCalculator(
                    exclude=("NMI", "AMI"), avg_of_avgs=avg_of_avgs, device=TEST_DEVICE
                )
                kwargs = {
                    "query_labels": query_labels,
                    "label_counts": label_counts,
                    "knn_labels": knn_labels,
                    "not_lone_query_mask": torch.ones(5, dtype=torch.bool)
                    if i == 0
                    else torch.zeros(5, dtype=torch.bool),
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
                        self.assertTrue(acc["mean_reciprocal_rank"] == 0)
                    else:
                        self.assertTrue(
                            isclose(
                                acc["precision_at_1"],
                                self.correct_precision_at_1(ecfss, avg_of_avgs),
                            )
                        )
                        self.assertTrue(
                            isclose(
                                acc["r_precision"],
                                self.correct_r_precision(ecfss, avg_of_avgs),
                            )
                        )
                        self.assertTrue(
                            isclose(
                                acc["mean_average_precision_at_r"],
                                self.correct_mean_average_precision_at_r(
                                    ecfss, avg_of_avgs
                                ),
                            )
                        )
                        self.assertTrue(
                            isclose(
                                acc["mean_average_precision"],
                                self.correct_mean_average_precision(ecfss, avg_of_avgs),
                            )
                        )
                        self.assertTrue(
                            isclose(
                                acc["mean_reciprocal_rank"],
                                self.correct_mean_reciprocal_rank(ecfss, avg_of_avgs),
                            )
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
            
    def correct_mean_reciprocal_rank(
        self, embeddings_come_from_same_source, avg_of_avgs
    ):
        # doesnt matter whether embeddings_come_from_same_source
        if not embeddings_come_from_same_source:
            acc0 = 1/2
            acc1 = 1
            acc2 = 1/5
            acc3 = 1
            acc4 = 1/3
        else:
            acc0 = 1
            acc1 = 1/2
            acc2 = 1/4
            acc3 = 1/2
            acc4 = 1/2

        if not avg_of_avgs:
            return np.mean([acc0, acc1, acc2, acc3, acc4])
        else:
            return np.mean([(acc0 + acc1) / 2, acc2, acc3, acc4])

    def test_get_lone_query_labels_custom(self):
        def fn1(x, y):
            return abs(x - y) < 2

        def fn2(x, y):
            return abs(x - y) > 99

        query_labels = torch.tensor(
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100], device=TEST_DEVICE
        )

        for comparison_fn in [fn1, fn2]:
            correct_unique_labels = torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100], device=TEST_DEVICE
            )

            if comparison_fn is fn1:
                correct_counts = torch.tensor(
                    [3, 4, 3, 3, 3, 3, 3, 3, 3, 2, 1], device=TEST_DEVICE
                )
                correct_lone_query_labels = torch.tensor([100], device=TEST_DEVICE)
                correct_not_lone_query_mask = torch.tensor(
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
                    ],
                    device=TEST_DEVICE,
                )
            elif comparison_fn is fn2:
                correct_counts = torch.tensor(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], device=TEST_DEVICE
                )
                correct_lone_query_labels = torch.tensor(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9], device=TEST_DEVICE
                )
                correct_not_lone_query_mask = torch.tensor(
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
                    ],
                    device=TEST_DEVICE,
                )

            label_counts = accuracy_calculator.get_label_match_counts(
                query_labels,
                query_labels,
                comparison_fn,
            )
            unique_labels, counts = label_counts

            self.assertTrue(torch.all(unique_labels == correct_unique_labels))
            self.assertTrue(torch.all(counts == correct_counts))

            (
                lone_query_labels,
                not_lone_query_mask,
            ) = accuracy_calculator.get_lone_query_labels(
                query_labels, label_counts, True, comparison_fn
            )

            self.assertTrue(torch.all(lone_query_labels == correct_lone_query_labels))
            self.assertTrue(
                torch.all(not_lone_query_mask == correct_not_lone_query_mask)
            )

    def test_get_lone_query_labels_multi_dim(self):
        def equality2D(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] == y[..., 1])

        def custom_label_comparison_fn(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

        query_labels = torch.tensor(
            [
                (1, 3),
                (0, 3),
                (0, 3),
                (0, 3),
                (1, 2),
                (4, 5),
            ],
            device=TEST_DEVICE,
        )

        for comparison_fn in [equality2D, custom_label_comparison_fn]:
            label_counts = accuracy_calculator.get_label_match_counts(
                query_labels,
                query_labels,
                comparison_fn,
            )

            unique_labels, counts = label_counts
            correct_unique_labels = torch.tensor(
                [[0, 3], [1, 2], [1, 3], [4, 5]], device=TEST_DEVICE
            )
            if comparison_fn is equality2D:
                correct_counts = torch.tensor([3, 1, 1, 1], device=TEST_DEVICE)
            else:
                correct_counts = torch.tensor([0, 1, 1, 0], device=TEST_DEVICE)

            self.assertTrue(torch.all(correct_counts == counts))
            self.assertTrue(torch.all(correct_unique_labels == unique_labels))

            if comparison_fn is equality2D:
                correct = [
                    (
                        True,
                        torch.tensor([[1, 2], [1, 3], [4, 5]], device=TEST_DEVICE),
                        torch.tensor(
                            [False, True, True, True, False, False], device=TEST_DEVICE
                        ),
                    ),
                    (
                        False,
                        torch.tensor([[]], device=TEST_DEVICE),
                        torch.tensor(
                            [True, True, True, True, True, True], device=TEST_DEVICE
                        ),
                    ),
                ]
            else:
                correct = [
                    (
                        True,
                        torch.tensor([[0, 3], [4, 5]], device=TEST_DEVICE),
                        torch.tensor(
                            [True, False, False, False, True, False], device=TEST_DEVICE
                        ),
                    ),
                    (
                        False,
                        torch.tensor([[0, 3], [4, 5]], device=TEST_DEVICE),
                        torch.tensor(
                            [True, False, False, False, True, False], device=TEST_DEVICE
                        ),
                    ),
                ]

            for same_source, correct_lone, correct_mask in correct:
                (
                    lone_query_labels,
                    not_lone_query_mask,
                ) = accuracy_calculator.get_lone_query_labels(
                    query_labels, label_counts, same_source, comparison_fn
                )
                if correct_lone.numel() == 0:
                    self.assertTrue(lone_query_labels.numel() == 0)
                else:
                    self.assertTrue(torch.all(lone_query_labels == correct_lone))

                self.assertTrue(torch.all(not_lone_query_mask == correct_mask))

    def test_get_lone_query_labels(self):
        query_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=TEST_DEVICE)
        reference_labels = torch.tensor([0, 0, 0, 1, 2, 2, 3, 4, 5], device=TEST_DEVICE)
        label_counts = accuracy_calculator.get_label_match_counts(
            query_labels,
            reference_labels,
            accuracy_calculator.EQUALITY,
        )

        for same_source, correct in [
            (True, torch.tensor([1, 3, 4, 5, 6], device=TEST_DEVICE)),
            (False, torch.tensor([6], device=TEST_DEVICE)),
        ]:
            lone_query_labels, _ = accuracy_calculator.get_lone_query_labels(
                query_labels,
                label_counts,
                same_source,
                accuracy_calculator.EQUALITY,
            )
            self.assertTrue(torch.all(lone_query_labels == correct))


class TestCalculateAccuraciesAndFaiss(unittest.TestCase):
    def test_accuracy_calculator_and_faiss_with_torch_and_numpy(self):
        for use_numpy in [True, False]:
            self._test_accuracy_calculator_and_faiss(use_numpy)
            self._test_accuracy_calculator_and_faiss_avg_of_avgs(use_numpy)
            self._test_accuracy_calculator_custom_comparison_function(use_numpy)
            self._test_accuracy_calculator_float_custom_comparison_function(use_numpy)

    def _test_accuracy_calculator_and_faiss(self, use_numpy):
        custom_knn = CustomKNN(LpDistance(normalize_embeddings=False, power=2))
        for knn_func in [None, custom_knn]:
            AC = accuracy_calculator.AccuracyCalculator(
                device=TEST_DEVICE, knn_func=knn_func
            )
            if use_numpy:
                query = np.arange(10)[:, None]
                reference = np.arange(10)[:, None]
                query_labels = np.arange(10)
                reference_labels = np.arange(10)
            else:
                query = torch.arange(10, device=TEST_DEVICE).unsqueeze(1)
                reference = torch.arange(10, device=TEST_DEVICE).unsqueeze(1)
                query_labels = torch.arange(10, device=TEST_DEVICE)
                reference_labels = torch.arange(10, device=TEST_DEVICE)
            acc = AC.get_accuracy(
                query, reference, query_labels, reference_labels, False
            )
            self.assertTrue(isclose(acc["precision_at_1"], 1))
            self.assertTrue(isclose(acc["r_precision"], 1))
            self.assertTrue(isclose(acc["mean_average_precision_at_r"], 1))

            if use_numpy:
                reference = (np.arange(20) / 2.0)[:, None]
                reference_labels = np.zeros(20)
                reference_labels[::2] = query_labels
                reference_labels[1::2] = np.ones(10)
            else:
                reference = (torch.arange(20, device=TEST_DEVICE) / 2.0).unsqueeze(1)
                reference_labels = torch.zeros(20, device=TEST_DEVICE)
                reference_labels[::2] = query_labels
                reference_labels[1::2] = torch.ones(10)
            acc = AC.get_accuracy(
                query, reference, query_labels, reference_labels, True
            )
            self.assertTrue(isclose(acc["precision_at_1"], 1))
            self.assertTrue(isclose(acc["r_precision"], 0.5))
            self.assertTrue(
                isclose(
                    acc["mean_average_precision_at_r"],
                    (1 + 2.0 / 2 + 3.0 / 5 + 4.0 / 7 + 5.0 / 9) / 10,
                )
            )

    def _test_accuracy_calculator_and_faiss_avg_of_avgs(self, use_numpy):
        AC_global_average = accuracy_calculator.AccuracyCalculator(
            avg_of_avgs=False, device=TEST_DEVICE
        )
        AC_per_class_average = accuracy_calculator.AccuracyCalculator(
            avg_of_avgs=True, device=TEST_DEVICE
        )

        query_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        reference_labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if use_numpy:
            query = np.arange(10)[:, None]
            reference = np.arange(10)[:, None]
            query_labels = np.array(query_labels)
            reference_labels = np.array(reference_labels)
        else:
            query = torch.arange(10, device=TEST_DEVICE).unsqueeze(1)
            reference = torch.arange(10, device=TEST_DEVICE).unsqueeze(1)
            query_labels = torch.tensor(query_labels, device=TEST_DEVICE)
            reference_labels = torch.tensor(reference_labels, device=TEST_DEVICE)

        query[-1] = 100
        reference[0] = -100

        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(isclose(acc["precision_at_1"], 0.9))
        self.assertTrue(isclose(acc["r_precision"], 0.9))
        self.assertTrue(isclose(acc["mean_average_precision_at_r"], 0.9))

        acc = AC_per_class_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(isclose(acc["precision_at_1"], 0.5))
        self.assertTrue(isclose(acc["r_precision"], 0.5))
        self.assertTrue(isclose(acc["mean_average_precision_at_r"], 0.5))

    def _test_accuracy_calculator_custom_comparison_function(self, use_numpy):
        def label_comparison_fn(x, y):
            return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

        self.assertRaises(
            NotImplementedError,
            lambda: accuracy_calculator.AccuracyCalculator(
                include=("NMI", "AMI"),
                avg_of_avgs=False,
                label_comparison_fn=label_comparison_fn,
                device=TEST_DEVICE,
            ),
        )

        AC_global_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=False,
            label_comparison_fn=label_comparison_fn,
            device=TEST_DEVICE,
        )

        AC_per_class_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=True,
            label_comparison_fn=label_comparison_fn,
            device=TEST_DEVICE,
        )

        query_labels = [
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

        reference_labels = [
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

        if use_numpy:
            query = np.arange(10)[:, None]
            reference = np.arange(10)[:, None]
            query_labels = np.array(query_labels)
            reference_labels = np.array(reference_labels)
        else:
            query = torch.arange(10, device=TEST_DEVICE).unsqueeze(1)
            reference = torch.arange(10, device=TEST_DEVICE).unsqueeze(1)
            query_labels = torch.tensor(query_labels, device=TEST_DEVICE)
            reference_labels = torch.tensor(reference_labels, device=TEST_DEVICE)

        query[-1] = 100
        reference[0] = -100

        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(isclose(acc["precision_at_1"], 0.9))
        self.assertTrue(isclose(acc["r_precision"], 0.9))
        self.assertTrue(isclose(acc["mean_average_precision_at_r"], 0.9))

        acc = AC_per_class_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )
        self.assertTrue(isclose(acc["precision_at_1"], 0.5))
        self.assertTrue(isclose(acc["r_precision"], 0.5))
        self.assertTrue(isclose(acc["mean_average_precision_at_r"], 0.5))

        query_labels = [
            (1, 3),
            (7, 3),
        ]

        reference_labels = [
            (1, 3),
            (7, 4),
            (1, 4),
            (1, 5),
            (1, 6),
        ]

        # SIMPLE CASE
        if use_numpy:
            query = np.arange(2)[:, None]
            reference = np.arange(5)[:, None]
            query_labels = np.array(query_labels)
            reference_labels = np.array(reference_labels)
        else:
            query = torch.arange(2, device=TEST_DEVICE).unsqueeze(1)
            reference = torch.arange(5, device=TEST_DEVICE).unsqueeze(1)
            query_labels = torch.tensor(query_labels, device=TEST_DEVICE)
            reference_labels = torch.tensor(reference_labels, device=TEST_DEVICE)

        correct = {
            "precision_at_1": 0.5,
            "r_precision": (1.0 / 3 + 1) / 2,
            "mean_average_precision_at_r": ((1.0 / 3) / 3 + 1) / 2,
        }

        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )

        for k in correct:
            self.assertTrue(isclose(acc[k], correct[k]))

        acc = AC_per_class_average.get_accuracy(
            query, reference, query_labels, reference_labels, False
        )

        for k in correct:
            self.assertTrue(isclose(acc[k], correct[k]))

    def _test_accuracy_calculator_float_custom_comparison_function(self, use_numpy):
        def label_comparison_fn(x, y):
            return torch.abs(x - y) < 1

        self.assertRaises(
            NotImplementedError,
            lambda: accuracy_calculator.AccuracyCalculator(
                include=("NMI", "AMI"),
                avg_of_avgs=False,
                label_comparison_fn=label_comparison_fn,
                device=TEST_DEVICE,
            ),
        )

        AC_global_average = accuracy_calculator.AccuracyCalculator(
            exclude=("NMI", "AMI"),
            avg_of_avgs=False,
            label_comparison_fn=label_comparison_fn,
            device=TEST_DEVICE,
        )

        query_labels = [
            0.01,
            0.02,
        ]

        reference_labels = [
            10.0,
            0.03,
            0.04,
            0.05,
        ]

        if use_numpy:
            query = np.array([0, 3])[:, None]
            reference = np.arange(4)[:, None]
            query_labels = np.array(query_labels)
            reference_labels = np.array(reference_labels)
        else:
            query = torch.tensor([0, 3], device=TEST_DEVICE).unsqueeze(1)
            reference = torch.arange(4, device=TEST_DEVICE).unsqueeze(1)
            query_labels = torch.tensor(
                query_labels,
                device=TEST_DEVICE,
            )
            reference_labels = torch.tensor(
                reference_labels,
                device=TEST_DEVICE,
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
            self.assertTrue(isclose(acc[k], correct[k]))

        correct = {
            "precision_at_1": 1.0,
            "r_precision": 1.0,
            "mean_average_precision_at_r": 1.0,
        }
        acc = AC_global_average.get_accuracy(
            query, reference, query_labels, reference_labels, True
        )
        for k in correct:
            self.assertTrue(isclose(acc[k], correct[k]))


class TestCalculateAccuraciesValidK(unittest.TestCase):
    def test_valid_k(self):
        for k in [-1, 0, 1.5, "max"]:
            self.assertRaises(
                ValueError, lambda: accuracy_calculator.AccuracyCalculator(k=k)
            )

    def test_k_warning(self):
        for k in [10, 100, 2000, "max_bin_count"]:
            AC = accuracy_calculator.AccuracyCalculator(k=k)
            embeddings = torch.randn(10000, 32)
            labels = torch.randint(0, 10, size=(10000,))
            level = "WARNING" if k in [10, 100] else "INFO"
            with self.assertLogs(level=level):
                AC.get_accuracy(embeddings, embeddings, labels, labels, True)


class TestCalculateAccuraciesFaissKNN(unittest.TestCase):
    def test_specify_gpu(self):
        max_world_size = min(4, torch.cuda.device_count())
        for i in range(1, max_world_size + 1):
            for gpus in itertools.combinations(range(max_world_size), i):
                knn_func = FaissKNN(gpus=gpus)
                AC = accuracy_calculator.AccuracyCalculator(knn_func=knn_func)
                embeddings = torch.randn(1000, 32)
                labels = torch.randint(0, 10, size=(1000,))
                AC.get_accuracy(embeddings, embeddings, labels, labels, True)

    def test_index_type(self):
        import faiss

        knn_func = FaissKNN(reset_after=False, index_init_fn=faiss.IndexFlatIP)
        AC = accuracy_calculator.AccuracyCalculator(knn_func=knn_func)
        embeddings = torch.randn(1000, 32)
        labels = torch.randint(0, 10, size=(1000,))
        AC.get_accuracy(embeddings, embeddings, labels, labels, True)
        self.assertTrue(isinstance(AC.knn_func.index, faiss.IndexFlatIP))


class TestCalculateAccuraciesCustomKNN(unittest.TestCase):
    def test_custom_knn(self):
        fn1 = CustomKNN(LpDistance(normalize_embeddings=False, power=2))
        fn2 = FaissKNN()
        AC1 = accuracy_calculator.AccuracyCalculator(knn_func=fn1)
        AC2 = accuracy_calculator.AccuracyCalculator(knn_func=fn2)
        embeddings = torch.randn(1000, 32)
        labels = torch.randint(0, 10, size=(1000,))
        acc1 = AC1.get_accuracy(embeddings, embeddings, labels, labels, True)
        acc2 = AC2.get_accuracy(embeddings, embeddings, labels, labels, True)

        for k, v in acc1.items():
            self.assertTrue(np.isclose(v, acc2[k], rtol=1e-3))
