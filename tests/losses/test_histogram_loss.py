import unittest

import torch
from numpy.testing import assert_almost_equal

from pytorch_metric_learning.losses import HistogramLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# DIRECTLY COPIED from https://github.com/valerystrizh/pytorch-histogram-loss/blob/master/losses.py.
# This code is copied from the official PyTorch implementation
# so that we can make sure our implementation returns the same result.
# Some minor changes were made to avoid errors during testing.
# Every change in the original code is reported and explained.
class OriginalImplementationHistogramLoss(torch.nn.Module):
    def __init__(self, num_steps, cuda=True):
        super(OriginalImplementationHistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.cuda = cuda
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t.cuda()

    def forward(self, features, classes):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (
                (s_repeat_floor - (self.t - self.step) > -self.eps)
                & (s_repeat_floor - (self.t - self.step) < self.eps)
                & inds
            )
            assert (
                indsa.nonzero().size()[0] == size
            ), "Another number of bins should be used"
            zeros = torch.zeros((1, indsa.size()[1])).byte()
            if self.cuda:
                zeros = zeros.cuda()
            indsb = torch.cat((indsa, zeros))[1:, :].to(
                dtype=torch.bool
            )  # Added to avoid bug with masks of uint8
            s_repeat_[~(indsb | indsa)] = 0
            # indsa corresponds to the first condition of the second equation of the paper
            self.t = self.t.to(
                dtype=s_repeat_.dtype
            )  # Added to avoid errors when using Half precision
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            # indsb corresponds to the second condition of the second equation of the paper
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size

        classes_size = classes.size()[0]
        classes_eq = (
            classes.repeat(classes_size, 1)
            == classes.view(-1, 1).repeat(1, classes_size)
        ).data
        dists = torch.mm(features, features.transpose(0, 1))
        assert (
            (dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item()
        ) == 0, "L2 normalization should be used"
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.cuda:
            s_inds = s_inds.cuda()
        classes_eq = classes_eq.to(
            device=s_inds.device
        )  # Added to avoid errors when using only cpu
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(
            histogram_pos.sum().item(),
            1,
            decimal=1,
            err_msg="Not good positive histogram",
            verbose=True,
        )
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(
            histogram_neg.sum().item(),
            1,
            decimal=1,
            err_msg="Not good negative histogram",
            verbose=True,
        )
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(
            1, histogram_pos.size()[0]
        )
        histogram_pos_inds = torch.tril(
            torch.ones(histogram_pos_repeat.size()), -1
        ).byte()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss


class TestHistogramLoss(unittest.TestCase):
    def test_histogram_loss(self):
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

            num_steps = 5 if dtype == torch.float16 else 21
            num_bins = num_steps - 1
            loss_func = HistogramLoss(n_bins=num_bins)

            loss = loss_func(embeddings, labels)

            original_loss_func = OriginalImplementationHistogramLoss(
                num_steps=num_steps, cuda=False
            )
            correct_loss = original_loss_func(embeddings, labels)

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

    def test_with_no_valid_triplets(self):
        loss_funcA = HistogramLoss(n_bins=4)
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
            lossA = loss_funcA(embeddings, labels)
            self.assertEqual(lossA, 0)

    def test_assertion_raises(self):
        with self.assertRaises(AssertionError):
            _ = HistogramLoss()

        with self.assertRaises(AssertionError):
            _ = HistogramLoss(n_bins=1, delta=0.5)

        with self.assertRaises(AssertionError):
            _ = HistogramLoss(n_bins=10, delta=0.4)
