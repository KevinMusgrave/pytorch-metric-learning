import unittest
from pytorch_metric_learning.utils import accuracy_calculator
import numpy as np

class TestCalculateAccuracies(unittest.TestCase):

    def test_accuracy_calculator(self):
        query_labels = np.array([0, 1, 2, 3, 4])
        knn_labels = np.array([[0, 1, 1, 2, 2],
                                    [1, 0, 1, 1, 3],
                                    [4, 4, 4, 4, 2],
                                    [3, 1, 3, 1, 3],
                                    [0, 0, 4, 2, 2]])
        label_counts = {0:2, 1:3, 2:5, 3:4, 4:5}
        AC = accuracy_calculator.AccuracyCalculator(exclude=("NMI", "AMI"))
        kwargs = {"query_labels": query_labels,
                "label_counts": label_counts,
                "knn_labels": knn_labels,
                "not_lone_query_idx": range(5)}

        function_dict = AC.get_function_dict()

        for ecfss in [False, True]:
            if ecfss:
                kwargs["knn_labels"] = kwargs["knn_labels"][:, 1:]
            kwargs["embeddings_come_from_same_source"] = ecfss
            acc = AC._get_accuracy(function_dict, **kwargs)
            self.assertTrue(acc["precision_at_1"]==self.correct_precision_at_1(ecfss))
            self.assertTrue(acc["r_precision"]==self.correct_r_precision(ecfss))
            self.assertTrue(acc["mean_average_precision_at_r"]==self.correct_mean_average_precision_at_r(ecfss))


    def correct_precision_at_1(self, embeddings_come_from_same_source):
        if not embeddings_come_from_same_source:
            return 0.6
        return 0
        
    def correct_r_precision(self, embeddings_come_from_same_source):
        if not embeddings_come_from_same_source:
            return np.mean([1./2, 2./3, 1./5, 2./4, 1./5])
        return np.mean([0./1, 1./2, 1./4, 1./3, 1./4])

    def correct_mean_average_precision_at_r(self, embeddings_come_from_same_source):
        if not embeddings_come_from_same_source:
            acc0 = (1) / 2
            acc1 = (1 + 2./3) / 3
            acc2 = (1./5) / 5
            acc3 = (1 + 2./3) / 4
            acc4 = (1./3) / 5
            return np.mean([acc0, acc1, acc2, acc3, acc4])
        else:
            acc0 = 0
            acc1 = (1./2) / 2
            acc2 = (1./4) / 4
            acc3 = (1./2) / 3
            acc4 = (1./2) / 4
            return np.mean([acc0, acc1, acc2, acc3, acc4])

    def test_get_label_counts(self):
        label_counts, num_k = accuracy_calculator.get_label_counts([0,1,3,2,3,1,3,3,4,6,5,10,4,4,4,4,6,6,5])
        self.assertTrue(label_counts=={0:1, 1:2, 2:1, 3:4, 4:5, 5:2, 6:3, 10:1})
        self.assertTrue(num_k==5)


    def test_get_lone_query_labels(self):
        query_labels = np.array([0, 1, 2, 3, 4, 5, 6])
        reference_labels = np.array([0,0,0,1,2,2,3,4,5,6])
        reference_label_counts, _ = accuracy_calculator.get_label_counts(reference_labels)

        lone_query_labels = accuracy_calculator.get_lone_query_labels(query_labels, reference_labels, reference_label_counts, True)
        self.assertTrue(np.all(np.unique(lone_query_labels) == np.array([1, 3, 4, 5, 6])))

        query_labels = np.array([0, 1, 2, 3, 4])
        reference_labels = np.array([0,0,0,1,2,2,4,5,6])

        lone_query_labels = accuracy_calculator.get_lone_query_labels(query_labels, reference_labels, reference_label_counts, False)
        self.assertTrue(np.all(np.unique(lone_query_labels) == np.array([3])))



class TestCalculateAccuraciesAndFaiss(unittest.TestCase):
    
    def test_accuracy_calculator_and_faiss(self):
        AC = accuracy_calculator.AccuracyCalculator(exclude=("NMI", "AMI"))

        query = np.arange(10)[:,None].astype(np.float32)
        reference = np.arange(10)[:,None].astype(np.float32)
        query_labels = np.arange(10).astype(np.int)
        reference_labels = np.arange(10).astype(np.int)
        acc = AC.get_accuracy(query, reference, query_labels, reference_labels, False)
        self.assertTrue(acc["precision_at_1"] == 1)
        self.assertTrue(acc["r_precision"] == 1)
        self.assertTrue(acc["mean_average_precision_at_r"] == 1)

        reference = (np.arange(20)/2.)[:,None].astype(np.float32)
        reference_labels = np.zeros(20).astype(np.int)
        reference_labels[::2] = query_labels
        reference_labels[1::2] = np.ones(10).astype(np.int)
        acc = AC.get_accuracy(query, reference, query_labels, reference_labels, True)
        self.assertTrue(acc["precision_at_1"] == 1)
        self.assertTrue(acc["r_precision"] == 0.5)
        self.assertTrue(acc["mean_average_precision_at_r"] == (1 + 2./2 + 3./5 + 4./7 + 5./9) / 10)


    def test_accuracy_calculator_and_faiss_average_per_class(self):
        AC_global_average = accuracy_calculator.AccuracyCalculator(exclude=("NMI", "AMI"), average_per_class=False)
        AC_per_class_average = accuracy_calculator.AccuracyCalculator(exclude=("NMI", "AMI"), average_per_class=True)
        query = np.arange(10)[:,None].astype(np.float32)
        reference = np.arange(10)[:,None].astype(np.float32)
        query[-1] = 100
        reference[0] = -100
        query_labels = np.array([0,0,0,0,0,0,0,0,0,1])
        reference_labels = np.array([1,0,0,0,0,0,0,0,0,0])
        acc = AC_global_average.get_accuracy(query, reference, query_labels, reference_labels, False)
        self.assertTrue(acc["precision_at_1"] == 0.9)
        self.assertTrue(acc["r_precision"] == 0.9)
        self.assertTrue(acc["mean_average_precision_at_r"] == 0.9)

        acc = AC_per_class_average.get_accuracy(query, reference, query_labels, reference_labels, False)
        self.assertTrue(acc["precision_at_1"] == 0.5)
        self.assertTrue(acc["r_precision"] == 0.5)
        self.assertTrue(acc["mean_average_precision_at_r"] == 0.5)


if __name__ == '__main__':
    unittest.main()