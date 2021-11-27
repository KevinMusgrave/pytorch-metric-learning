import unittest

import numpy as np
import torch

from pytorch_metric_learning.utils import accuracy_calculator, stat_utils

from .. import TEST_DEVICE

### FROM https://gist.github.com/VChristlein/fd55016f8d1b38e95011a025cbff9ccc
### and https://github.com/KevinMusgrave/pytorch-metric-learning/issues/290


class TestCalculateAccuraciesLargeK(unittest.TestCase):
    def test_accuracy_calculator_large_k(self):
        for ecfss in [False, True]:
            for max_k in [None, "max_bin_count"]:
                for num_embeddings in [1000, 2100]:
                    # make random features
                    encs = np.random.rand(num_embeddings, 5).astype(np.float32)
                    # and random labels of 100 classes
                    labels = np.zeros((num_embeddings // 100, 100), dtype=int)
                    for i in range(10):
                        labels[i] = np.arange(100)
                    labels = labels.ravel()

                    correct_p1, correct_map, correct_mapr = self.evaluate(
                        encs, labels, max_k, ecfss
                    )

                    # use Musgrave's library
                    if max_k is None:
                        k = len(encs) - 1 if ecfss else len(encs)
                        accs = [
                            accuracy_calculator.AccuracyCalculator(device=TEST_DEVICE),
                            accuracy_calculator.AccuracyCalculator(
                                k=k, device=TEST_DEVICE
                            ),
                        ]
                    elif max_k == "max_bin_count":
                        accs = [
                            accuracy_calculator.AccuracyCalculator(
                                k="max_bin_count", device=TEST_DEVICE
                            )
                        ]

                    for acc in accs:
                        d = acc.get_accuracy(
                            encs,
                            encs,
                            labels,
                            labels,
                            ecfss,
                            include=(
                                "mean_average_precision",
                                "mean_average_precision_at_r",
                                "precision_at_1",
                            ),
                        )

                        self.assertTrue(np.isclose(correct_p1, d["precision_at_1"]))
                        self.assertTrue(
                            np.isclose(correct_map, d["mean_average_precision"])
                        )
                        self.assertTrue(
                            np.isclose(correct_mapr, d["mean_average_precision_at_r"])
                        )

    def evaluate(self, encs, labels, max_k=None, ecfss=False):
        """
        evaluate encodings assuming using associated labels
        parameters:
            encs: TxD encoding matrix
            labels: array/list of T labels
        """

        # let's use Musgrave's knn
        torch_encs = torch.from_numpy(encs)
        k = len(encs) - 1 if ecfss else len(encs)
        all_indices, _ = stat_utils.get_knn(torch_encs, torch_encs, k, ecfss)
        if max_k is None:
            max_k = k
            indices = all_indices
        elif max_k == "max_bin_count":
            max_k = int(max(np.bincount(labels))) - int(ecfss)
            indices, _ = stat_utils.get_knn(torch_encs, torch_encs, max_k, ecfss)

        # let's use the most simple mAP implementation
        # of course this can be computed much faster using cumsum, etc.
        n_encs = len(encs)
        mAP = []
        mAP_at_r = []
        correct = 0
        for r in range(n_encs):
            precisions = []
            rel = 0
            # indices doesn't contain the query index itself anymore, so no correction w. -1 necessary
            all_rel = np.count_nonzero(labels[all_indices[r]] == labels[r])
            prec_at_r = []
            for k in range(max_k):
                if labels[indices[r, k]] == labels[r]:
                    rel += 1
                    precisions.append(rel / float(k + 1))
                    if k == 0:
                        correct += 1

                    # mAP@R
                    if k < all_rel:
                        prec_at_r.append(rel / float(k + 1))

            avg_precision = np.mean(precisions) if len(precisions) > 0 else 0
            mAP.append(avg_precision)
            # mAP@R
            avg_prec_at_r = np.sum(prec_at_r) / all_rel if all_rel > 0 else 0
            mAP_at_r.append(avg_prec_at_r)

        mAP = np.mean(mAP)
        mAP_at_r = np.mean(mAP_at_r)
        return float(correct) / n_encs, mAP, mAP_at_r
