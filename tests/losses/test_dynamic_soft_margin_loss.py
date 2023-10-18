import torch
import torch.nn as nn

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# DIRECTLY COPIED FROM https://github.com/lg-zhang/dynamic-soft-margin-pytorch/blob/master/modules/dynamic_soft_margin.py.
# This code is copied from the official implementation
# so that we can make sure our implementation returns the same result.
# Some minor changes were made to avoid errors during testing.
# Every change in the original code is reported and explained.
def compute_distance_matrix_unit_l2(a, b, eps=1e-6):
    """
    computes pairwise Euclidean distance and return a N x N matrix
    """

    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = ((1.0 - dmat + eps) * 2.0).pow(0.5)
    return dmat


def find_hard_negatives(dmat, output_index=True, empirical_thresh=0.0):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...
    """

    r, c = dmat.size()  # Correct bug

    if not output_index:
        pos = torch.zeros(max(r, c))  # Correct bug
        pos[: min(r, c)] = dmat.diag()  # Correct bug

    dmat = (
        dmat + torch.eye(r, c).to(dmat.device) * 99999
    )  # filter diagonal     # Correct bug
    dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset

    # Add following 3 lines to solve a bug
    min_a, min_p = torch.zeros(max(r, c)), torch.zeros(
        max(r, c)
    )  # Check for unequal number of anchors and positives
    min_a[:c], _ = torch.min(dmat, dim=0)
    min_p[:r], _ = torch.min(dmat, dim=1)

    if not output_index:
        neg = torch.min(min_a, min_p)
        return pos, neg.to(dtype=pos.dtype)  # Added cast to avoid errors

    # Useless for our testing purposes
    # mask = min_a < min_p
    # a_idx = torch.cat(
    #     (mask.nonzero().view(-1) + cnt, (~mask).nonzero().view(-1))
    # )  # use p as anchor
    # p_idx = torch.cat(
    #     (mask.nonzero().view(-1), (~mask).nonzero().view(-1) + cnt)
    # )  # use a as anchor
    # n_idx = torch.cat((min_a_idx[mask], min_p_idx[~mask] + cnt))
    # return a_idx, p_idx, n_idx


class OriginalImplementationDynamicSoftMarginLoss(nn.Module):
    def __init__(self, is_binary=False, momentum=0.01, max_dist=None, nbins=512):
        """
        is_binary: true if learning binary descriptor
        momentum: weight assigned to the histogram computed from the current batch
        max_dist: maximum possible distance in the feature space
        nbins: number of bins to discretize the PDF
        """
        super(OriginalImplementationDynamicSoftMarginLoss, self).__init__()
        self._is_binary = is_binary

        if max_dist is None:
            # max_dist = 256 if self._is_binary else 2.0
            max_dist = 2.0

        self._momentum = momentum
        self._max_val = max_dist
        self._min_val = -max_dist
        self.register_buffer("histogram", torch.ones(nbins))

        self._stats_initialized = False
        self.current_step = None

    def _compute_distances(self, x, labels=None):
        # Useless for testing purposes
        # if self._is_binary:
        #     return self._compute_hamming_distances(x)
        # else:
        return self._compute_l2_distances(x, labels=labels)

    # Formatted to test with and without labels
    def _compute_l2_distances(self, x, labels=None):
        if labels is None:
            cnt = x.size(0) // 2
            a = x[:cnt, :]
            p = x[cnt:, :]
            dmat = compute_distance_matrix_unit_l2(a, p)
            return find_hard_negatives(dmat, output_index=False, empirical_thresh=0.008)
        else:
            dmat = compute_distance_matrix_unit_l2(x, x)
            dmat.fill_diagonal_(0)  # Put distance to itself to 0
            anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(
                None, labels, labels, t_per_anchor="all"
            )
            return dmat[anchor_idx, positive_idx], dmat[anchor_idx, negative_idx]

    # We do not use binary descriptors
    # def _compute_hamming_distances(self, x):
    #     cnt = x.size(0) // 2
    #     ndims = x.size(1)
    #     a = x[:cnt, :]
    #     p = x[cnt:, :]

    #     dmat = compute_distance_matrix_hamming(
    #         (a > 0).float() * 2.0 - 1.0, (p > 0).float() * 2.0 - 1.0
    #     )
    #     a_idx, p_idx, n_idx = find_hard_negatives(
    #         dmat, output_index=True, empirical_thresh=2
    #     )

    #     # differentiable Hamming distance
    #     a = x[a_idx, :]
    #     p = x[p_idx, :]
    #     n = x[n_idx, :]

    #     pos_dist = (1.0 - a * p).sum(dim=1) / ndims
    #     neg_dist = (1.0 - a * n).sum(dim=1) / ndims

    #     # non-differentiable Hamming distance
    #     a_b = (a > 0).float() * 2.0 - 1.0
    #     p_b = (p > 0).float() * 2.0 - 1.0
    #     n_b = (n > 0).float() * 2.0 - 1.0

    #     pos_dist_b = (1.0 - a_b * p_b).sum(dim=1) / ndims
    #     neg_dist_b = (1.0 - a_b * n_b).sum(dim=1) / ndims

    #     return pos_dist, neg_dist, pos_dist_b, neg_dist_b

    def _compute_histogram(self, x, momentum):
        """
        update the histogram using the current batch
        """
        num_bins = self.histogram.size(0)
        x_detached = x.detach()
        self.bin_width = (self._max_val - self._min_val) / num_bins  # Adjusted formula
        lo = torch.floor(
            (x_detached - self._min_val) / self.bin_width
        ).long()  # Add cast to avoid errors
        hi = (lo + 1).clamp(min=0, max=num_bins - 1)
        hist = x.new_zeros(num_bins)
        alpha = (
            1.0
            - (x_detached - self._min_val - lo.float() * self.bin_width)
            / self.bin_width
        ).to(
            dtype=hist.dtype
        )  # Added cast to avoid errors
        hist.index_add_(0, lo, alpha)
        hist.index_add_(0, hi, 1.0 - alpha)
        hist = hist / (hist.sum() + 1e-6)
        self.histogram = c_f.to_device(
            self.histogram, tensor=hist, dtype=hist.dtype
        )  # Line added to avoid errors
        self.histogram = (1.0 - momentum) * self.histogram + momentum * hist

    def _compute_stats(self, pos_dist, neg_dist):
        hist_val = pos_dist - neg_dist
        if self._stats_initialized:
            self._compute_histogram(hist_val, self._momentum)
        else:
            self._compute_histogram(hist_val, 1.0)
            self._stats_initialized = True

    def forward(self, x, labels=None):
        distances = self._compute_distances(x, labels=labels)
        if not self._is_binary:
            pos_dist, neg_dist = distances
            self._compute_stats(pos_dist, neg_dist)
            hist_var = pos_dist - neg_dist
        else:
            pos_dist, neg_dist, pos_dist_b, neg_dist_b = distances
            self._compute_stats(pos_dist_b, neg_dist_b)
            hist_var = pos_dist_b - neg_dist_b

        PDF = self.histogram / self.histogram.sum()
        CDF = PDF.cumsum(0)

        # lookup weight from the CDF
        bin_idx = torch.floor((hist_var - self._min_val) / self.bin_width).long()
        weight = CDF[bin_idx]

        # Changed to an equivalent version for making same computation as in dynamic_soft_margin_loss.py
        # loss = -(neg_dist * weight).mean() + (pos_dist * weight).mean()
        loss = (hist_var*weight).mean()
        return loss.to(device=x.device, dtype=x.dtype)  # Added cast to avoid errors


import unittest

from pytorch_metric_learning.losses import DynamicSoftMarginLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestDynamicSoftMarginLoss(unittest.TestCase):
    def test_dynamic_soft_margin_loss_without_labels(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            embeddings = torch.nn.functional.normalize(embeddings)
            cnt = embeddings.size(0) // 2

            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-3.0,
                num_bins=10,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-3.0,
                num_bins=20,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-3.0,
                num_bins=30,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-2.0,
                num_bins=10,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-2.0,
                num_bins=20,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-2.0,
                num_bins=30,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-1.0,
                num_bins=10,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-1.0,
                num_bins=20,
            )
            self.helper(
                embeddings[:cnt, :],
                None,
                dtype,
                ref_emb=embeddings[cnt:, :],
                min_val=-1.0,
                num_bins=30,
            )

    def test_dynamic_soft_margin_loss_with_labels(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            embeddings = torch.nn.functional.normalize(embeddings)
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            self.helper(embeddings, labels, dtype, min_val=-3.0, num_bins=10)
            self.helper(embeddings, labels, dtype, min_val=-3.0, num_bins=20)
            self.helper(embeddings, labels, dtype, min_val=-3.0, num_bins=30)
            self.helper(embeddings, labels, dtype, min_val=-2.0, num_bins=10)
            self.helper(embeddings, labels, dtype, min_val=-2.0, num_bins=20)
            self.helper(embeddings, labels, dtype, min_val=-2.0, num_bins=30)
            self.helper(embeddings, labels, dtype, min_val=-1.0, num_bins=10)
            self.helper(embeddings, labels, dtype, min_val=-1.0, num_bins=20)
            self.helper(embeddings, labels, dtype, min_val=-1.0, num_bins=30)

    def helper(
        self,
        embeddings,
        labels,
        dtype,
        ref_emb=None,
        ref_labels=None,
        min_val=-2.0,
        num_bins=10,
    ):
        loss_func = DynamicSoftMarginLoss(min_val=min_val, num_bins=num_bins)
        original_loss_func = OriginalImplementationDynamicSoftMarginLoss(
            max_dist=-min_val, nbins=num_bins
        )

        loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
        if labels is None:
            embeddings = torch.cat((embeddings, ref_emb))
        correct_loss = original_loss_func(embeddings, labels)

        rtol = 1e-2 if dtype == torch.float16 else 1e-5
        self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

    def test_with_no_valid_triplets(self):
        loss_func = DynamicSoftMarginLoss()
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 1, 2, 3, 4])
            loss = loss_func(embeddings, labels)
            self.assertEqual(loss, 0)

    def test_backward(self):
        for dtype in TEST_DTYPES:
            loss_func = DynamicSoftMarginLoss()
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            loss = loss_func(embeddings, labels)
            loss.backward()
