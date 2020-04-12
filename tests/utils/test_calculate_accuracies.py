import unittest
from pytorch_metric_learning.utils import calculate_accuracies
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
        accuracy_calculator = calculate_accuracies.AccuracyCalculator()
        kwargs = {"query_labels": query_labels,
                "label_counts": label_counts,
                "knn_labels": knn_labels}

        function_dict = accuracy_calculator.get_function_dict()
        function_dict.pop("NMI", None)

        for ecfss in [False, True]:
            if ecfss:
                kwargs["knn_labels"] = kwargs["knn_labels"][:, 1:]
            kwargs["embeddings_come_from_same_source"] = ecfss
            acc = accuracy_calculator._get_accuracy(function_dict, **kwargs)
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
        label_counts, num_k = calculate_accuracies.get_label_counts([0,1,3,2,3,1,3,3,4,6,5,10,4,4,4,4,6,6,5])
        self.assertTrue(label_counts=={0:1, 1:2, 2:1, 3:4, 4:5, 5:2, 6:3, 10:1})
        self.assertTrue(num_k==5)


if __name__ == '__main__':
    unittest.main()