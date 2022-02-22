import unittest
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SubCenterArcFaceLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestSubCenterArcFaceLoss(unittest.TestCase):
    def test_subcenter_arcface_loss(self):
        pass
