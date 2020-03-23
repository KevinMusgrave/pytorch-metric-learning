import unittest
from pytorch_metric_learning.utils import calculate_accuracies
import numpy as np

class TestCalculateAccuracies(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.query_labels = np.array([0, 1, 2, 3, 4])
        self.knn_labels = np.array([[0, 1, 1, 2, 2],
                                    [1, 0, 1, 1, 3],
                                    [4, 4, 4, 4, 2],
                                    [3, 1, 3, 1, 3],
                                    [0, 0, 4, 2, 2]])
        self.label_counts = {0:2, 1:3, 2:5, 3:4, 4:5}

    def test_precision_at_1(self):
        acc = calculate_accuracies.precision_at_k(self.knn_labels, self.query_labels[:, None], 1)
        self.assertTrue(acc==0.6)

    def test_r_precision(self):
        acc = calculate_accuracies.r_precision(self.knn_labels, self.query_labels[:, None], False, self.label_counts)
        correct_acc = np.mean([1./2, 2./3, 1./5, 2./4, 1./5])
        self.assertTrue(acc==correct_acc)

        acc = calculate_accuracies.r_precision(self.knn_labels[:,1:], self.query_labels[:, None], True, self.label_counts)
        correct_acc = np.mean([0./1, 1./2, 1./4, 1./3, 1./4])
        self.assertTrue(acc==correct_acc)

    def test_mean_average_precision_at_r(self):
        acc = calculate_accuracies.mean_average_precision_at_r(self.knn_labels, self.query_labels[:, None], False, self.label_counts)
        acc0 = (1) / 2
        acc1 = (1 + 2./3) / 3
        acc2 = (1./5) / 5
        acc3 = (1 + 2./3) / 4
        acc4 = (1./3) / 5
        correct_acc = np.mean([acc0, acc1, acc2, acc3, acc4])
        self.assertTrue(acc==correct_acc)

        acc = calculate_accuracies.mean_average_precision_at_r(self.knn_labels[:,1:], self.query_labels[:, None], True, self.label_counts)
        acc0 = 0
        acc1 = (1./2) / 2
        acc2 = (1./4) / 4
        acc3 = (1./2) / 3
        acc4 = (1./2) / 4
        correct_acc = np.mean([acc0, acc1, acc2, acc3, acc4])
        self.assertTrue(acc==correct_acc)

    def test_get_label_counts(self):
        label_counts, num_k = calculate_accuracies.get_label_counts([0,1,3,2,3,1,3,3,4,6,5,10,4,4,4,4,6,6,5])
        self.assertTrue(label_counts=={0:1, 1:2, 2:1, 3:4, 4:5, 5:2, 6:3, 10:1})
        self.assertTrue(num_k==5)


if __name__ == '__main__':
    unittest.main()