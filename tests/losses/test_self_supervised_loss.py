import unittest

import torch

import pytorch_metric_learning.losses as losses

from .. import TEST_DEVICE, TEST_DTYPES


class TestSelfSupervisedLoss(unittest.TestCase):
    def test_ssl_wrapper_all(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                100,
                256,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            embeddings = torch.nn.functional.normalize(embeddings)

            ref_emb = torch.randn(
                100,
                256,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            ref_emb = torch.nn.functional.normalize(ref_emb)

            labels = torch.arange(100).to(TEST_DEVICE)

            real_losses = self.run_all_loss_fns(embeddings, labels, ref_emb, labels)
            losses = self.run_all_loss_fns_wrapped(embeddings, ref_emb)

            atol = 1e-3
            for loss_name, loss_value in losses.items():
                self.assertTrue(
                    torch.isclose(real_losses[loss_name], loss_value, atol=atol)
                )

    def run_all_loss_fns(self, embeddings, labels, ref_emb, ref_labels):
        loss_fns_list = self.load_valid_loss_fns()

        loss_fns = dict()
        for loss_fn in loss_fns_list:
            loss_name = type(loss_fn).__name__
            loss_value = loss_fn(
                embeddings=embeddings,
                labels=labels,
                ref_emb=ref_emb,
                ref_labels=ref_labels,
            )
            loss_fns[loss_name] = loss_value

        return loss_fns

    def run_all_loss_fns_wrapped(self, embeddings, ref_emb):
        loss_fns_list = self.load_valid_loss_fns()

        loss_fns = dict()
        for loss_fn in loss_fns_list:
            loss_name = type(loss_fn).__name__
            loss_fn = losses.SelfSupervisedLoss(loss_fn)
            loss_value = loss_fn(embeddings=embeddings, ref_emb=ref_emb)
            loss_fns[loss_name] = loss_value

        return loss_fns

    def load_valid_loss_fns(self):
        supported_losses = losses.SelfSupervisedLoss.supported_losses()

        loss_fns = [
            losses.AngularLoss(),
            losses.CircleLoss(),
            losses.ContrastiveLoss(),
            losses.GeneralizedLiftedStructureLoss(),
            losses.IntraPairVarianceLoss(),
            losses.LiftedStructureLoss(),
            losses.MultiSimilarityLoss(),
            losses.NTXentLoss(),
            losses.SignalToNoiseRatioContrastiveLoss(),
            losses.SupConLoss(),
            losses.TripletMarginLoss(),
            losses.NCALoss(),
            losses.TupletMarginLoss(),
        ]

        loaded_loss_names = [type(loss).__name__ for loss in loss_fns]
        assert set(loaded_loss_names).intersection(set(supported_losses)) == set(
            supported_losses
        )

        return loss_fns
