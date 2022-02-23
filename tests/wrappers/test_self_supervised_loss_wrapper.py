import inspect
import unittest

import numpy as np
import torch

import pytorch_metric_learning.losses as losses
from pytorch_metric_learning.wrappers import SelfSupervisedLossWrapper

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestSelfSupervisedLossWrapper(unittest.TestCase):
    def test_ssl_wrapper_same(self):
        """
        Test for distance ~ 0 when given same input embeddings
        for distance metrics where this is true.

        Other distance metrics that are in supported_loses() not included here (commented)
        have caveats that prevent it from reaching 0 values.

        Another thing to note is distance -> 0 is less true the
            1. smaller the embedding sizes are
            2. the less embeddings there are to consider.
        This is why the test here uses 200 embeddings of size 512 and atol=1e-3
        """
        for dtype in TEST_DTYPES:
            loss_fns = [
                SelfSupervisedLossWrapper(losses.AngularLoss()),
                SelfSupervisedLossWrapper(losses.CircleLoss()),
                SelfSupervisedLossWrapper(losses.ContrastiveLoss()),
                # SelfSupervisedLossWrapper(losses.GeneralizedLiftedStructureLoss()),
                SelfSupervisedLossWrapper(losses.IntraPairVarianceLoss()),
                SelfSupervisedLossWrapper(losses.LiftedStructureLoss()),
                SelfSupervisedLossWrapper(losses.MultiSimilarityLoss()),
                SelfSupervisedLossWrapper(losses.NTXentLoss()),
                SelfSupervisedLossWrapper(losses.SignalToNoiseRatioContrastiveLoss()),
                SelfSupervisedLossWrapper(losses.SupConLoss()),
                SelfSupervisedLossWrapper(losses.TripletMarginLoss()),
                # SelfSupervisedLossWrapper(losses.NCALoss()),
                SelfSupervisedLossWrapper(losses.TupletMarginLoss()),
            ]
            embeddings = torch.randn(
                300,
                1024,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            embeddings = torch.nn.functional.normalize(embeddings)
            ref_emb = embeddings
            zero = torch.tensor(0.0, dtype=dtype).to(TEST_DEVICE)

            atol = 1e-2
            for loss_fn in loss_fns:
                loss = loss_fn(embeddings, ref_emb)
                self.assertTrue(torch.isclose(zero, loss, atol=atol))

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
            loss_fn = SelfSupervisedLossWrapper(loss_fn)
            loss_value = loss_fn(embeddings=embeddings, ref_emb=ref_emb)
            loss_fns[loss_name] = loss_value

        return loss_fns

    def load_valid_loss_fns(self):
        reqparams = ["embeddings", "labels", "ref_emb", "ref_labels"]
        supported_losses = SelfSupervisedLossWrapper.supported_losses()

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

        # loss_names = [
        #     loss for loss in dir(losses)
        #         if not (loss.startswith("__") or loss.endswith("__"))
        #         and (loss[0].isupper())
        #         and loss not in BLACKLIST
        # ]

        # loss_fns = dict()
        # for module_name in loss_names:
        #     loss_fn = getattr(losses, module_name)
        #     '''
        #     1. if the loss_fn does not support ref_emb or ref_labels
        #     '''
        #     if "compute_loss" not in dir(loss_fn):
        #         continue

        #     args = inspect.getfullargspec(loss_fn.compute_loss).args
        #     if len(set(args).intersection(set(REQPARAMS))) != 4:
        #         continue

        #     '''
        #     2. loss function does not have full default values for its parameters. ([1:] for "self")
        #     '''
        #     initparams = inspect.getfullargspec(loss_fn.__init__)
        #     if initparams.defaults is None or len(initparams.args[1:]) != len(initparams.defaults):
        #         continue

        #     loss_fns[module_name] = loss_fn()

        return loss_fns
