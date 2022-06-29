import unittest

import torch

from pytorch_metric_learning import losses
from pytorch_metric_learning.miners import MultiSimilarityMiner, TripletMarginMiner
from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_pairs_indices

from .. import TEST_DEVICE


def get_compatible_losses():
    return [
        losses.CircleLoss(),
        losses.ContrastiveLoss(),
        losses.IntraPairVarianceLoss(),
        losses.GeneralizedLiftedStructureLoss(),
        losses.LiftedStructureLoss(),
        losses.MarginLoss(),
        losses.MultiSimilarityLoss(),
        losses.NTXentLoss(),
        losses.SignalToNoiseRatioContrastiveLoss(),
        losses.SupConLoss(),
        losses.TripletMarginLoss(),
        losses.TupletMarginLoss(),
    ]


def get_incompatible_losses():
    return [
        losses.AngularLoss(),
        losses.ArcFaceLoss(10, 32),
        losses.CosFaceLoss(10, 32),
        losses.FastAPLoss(),
        losses.InstanceLoss(),
        losses.LargeMarginSoftmaxLoss(10, 32),
        losses.NPairsLoss(),
        losses.NCALoss(),
        losses.NormalizedSoftmaxLoss(10, 32),
        losses.ProxyAnchorLoss(10, 32),
        losses.ProxyNCALoss(10, 32),
        losses.SoftTripleLoss(10, 32),
        losses.SphereFaceLoss(10, 32),
        losses.SubCenterArcFaceLoss(num_classes=10, embedding_size=32),
    ]


class TestLossesWithoutLabels(unittest.TestCase):
    def test(self):
        emb = torch.randn(64, 32, device=TEST_DEVICE)
        labels = torch.randint(0, 10, size=(64,), device=TEST_DEVICE)
        pairs1 = get_all_pairs_indices(labels)
        pairs2 = MultiSimilarityMiner()(emb, labels)
        pairs3 = TripletMarginMiner()(emb, labels)
        compatible_losses = get_compatible_losses()
        incompatible_losses = get_incompatible_losses()

        for p in [pairs1, pairs2, pairs3]:
            for loss_fn in compatible_losses:
                loss_with_labels = loss_fn(emb, labels, p)
                loss_without_labels1 = loss_fn(emb, indices_tuple=p)
                loss_without_labels2 = loss_fn(emb, indices_tuple=p, ref_emb=emb)
                self.assertEqual(loss_with_labels.item(), loss_without_labels1.item())
                self.assertEqual(loss_with_labels.item(), loss_without_labels2.item())

            for loss_fn in compatible_losses:
                with self.assertRaises(ValueError):
                    loss_fn(emb)
                with self.assertRaises(ValueError):
                    loss_fn(emb, ref_emb=emb)

            for loss_fn in incompatible_losses:
                loss_fn(emb, labels)  # should work
                with self.assertRaises(ValueError):
                    loss_fn(emb, indices_tuple=p)
