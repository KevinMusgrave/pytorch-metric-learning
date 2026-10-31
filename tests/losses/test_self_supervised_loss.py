import unittest
from functools import partial

import torch

from pytorch_metric_learning import losses

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def ntxent_reference(*views, symmetric, temperature=0.07):
    num_samples = len(views[0])
    unit = torch.nn.functional.normalize(torch.cat(views), dim=1)
    labels = torch.arange(num_samples, device=unit.device).repeat(len(views))
    anchors = range(len(unit)) if symmetric else range(num_samples)
    references = range(len(unit)) if symmetric else range(num_samples, len(unit))
    terms = []
    for a in anchors:
        negatives = [r for r in references if r != a and labels[r] != labels[a]]
        negative_sum = sum(
            torch.exp(unit[a].dot(unit[n]) / temperature) for n in negatives
        )
        for p in references:
            if p != a and labels[p] == labels[a]:
                positive = torch.exp(unit[a].dot(unit[p]) / temperature)
                terms.append(-torch.log(positive / (positive + negative_sum)))
    return torch.stack(terms).mean(), len(terms)


def reference_loss(*views, symmetric):
    return ntxent_reference(*views, symmetric=symmetric)[0]


class TestSelfSupervisedLoss(unittest.TestCase):
    def test_more_than_two_views_match_scalar_reference(self):
        torch.manual_seed(5)
        views = [torch.randn(3, 4) for _ in range(4)]
        expected_pairs = {True: 4 * 3 * 3, False: 3 * 3}
        for symmetric in (True, False):
            loss_func = losses.SelfSupervisedLoss(
                losses.NTXentLoss(), symmetric=symmetric
            )
            reference = partial(reference_loss, symmetric=symmetric)
            pairs = ntxent_reference(*views, symmetric=symmetric)[1]
            self.assertEqual(pairs, expected_pairs[symmetric])
            for dtype in TEST_DTYPES:
                with self.subTest(symmetric=symmetric, dtype=dtype):
                    helpers.assert_matches_float64(loss_func, reference, views, dtype)

    def test_every_view_receives_gradients(self):
        for symmetric in (True, False):
            views = [
                torch.randn(4, 3, requires_grad=True, device=TEST_DEVICE)
                for _ in range(3)
            ]
            losses.SelfSupervisedLoss(losses.SupConLoss(), symmetric=symmetric)(
                *views
            ).backward()
            for view in views:
                self.assertGreater(torch.as_tensor(view.grad).abs().sum().item(), 0)

    def test_rejects_views_of_different_shapes(self):
        loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss())
        view = torch.randn(4, 3, device=TEST_DEVICE)
        for other in (view[:3], view[:, :2]):
            with self.assertRaisesRegex(ValueError, "same shape"):
                loss_func(view, view, other)

    def test_ssl_wrapper_all(self):
        for dtype in TEST_DTYPES:
            for symmetric in [True, False]:
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

                real_losses = self.run_all_loss_fns(
                    embeddings, labels, ref_emb, labels.clone(), symmetric
                )
                losses = self.run_all_loss_fns_wrapped(embeddings, ref_emb, symmetric)

                atol = 1e-3
                for loss_name, loss_value in losses.items():
                    self.assertTrue(
                        torch.isclose(real_losses[loss_name], loss_value, atol=atol)
                    )

    def run_all_loss_fns(self, embeddings, labels, ref_emb, ref_labels, symmetric):
        if symmetric:
            embeddings = torch.cat([embeddings, ref_emb], dim=0)
            labels = torch.cat([labels, ref_labels], dim=0)
        loss_fns_list = self.load_valid_loss_fns()

        loss_fns = {}
        for loss_fn in loss_fns_list:
            loss_name = type(loss_fn).__name__
            if symmetric:
                loss_value = loss_fn(embeddings, labels)
            else:
                loss_value = loss_fn(
                    embeddings=embeddings,
                    labels=labels,
                    ref_emb=ref_emb,
                    ref_labels=ref_labels,
                )
            loss_fns[loss_name] = loss_value

        return loss_fns

    def run_all_loss_fns_wrapped(self, embeddings, ref_emb, symmetric):
        loss_fns_list = self.load_valid_loss_fns()

        loss_fns = {}
        for loss_fn in loss_fns_list:
            loss_name = type(loss_fn).__name__
            loss_fn = losses.SelfSupervisedLoss(loss_fn, symmetric=symmetric)
            loss_value = loss_fn(embeddings=embeddings, ref_emb=ref_emb)
            loss_fns[loss_name] = loss_value

        return loss_fns

    def load_valid_loss_fns(self):
        loss_fns = [
            losses.AngularLoss(),
            losses.CircleLoss(),
            losses.ContrastiveLoss(),
            losses.GeneralizedLiftedStructureLoss(),
            losses.IntraPairVarianceLoss(),
            losses.MultiSimilarityLoss(),
            losses.NTXentLoss(),
            losses.SignalToNoiseRatioContrastiveLoss(),
            losses.SupConLoss(),
            losses.TripletMarginLoss(),
            losses.NCALoss(),
            losses.TupletMarginLoss(),
        ]

        return loss_fns
