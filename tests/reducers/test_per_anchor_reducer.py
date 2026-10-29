import unittest

import torch

from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    MeanReducer,
    PerAnchorReducer,
)
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES


def per_anchor_reference(losses, anchors, aggregation, inner):
    values = {}
    for loss, anchor in zip(losses.double().tolist(), anchors.tolist(), strict=True):
        values.setdefault(anchor, []).append(loss)
    reduce = (lambda x: sum(x) / len(x)) if aggregation == "mean" else sum
    per_anchor = [reduce(values[anchor]) for anchor in sorted(values)]
    if isinstance(inner, AvgNonZeroReducer):
        per_anchor = [x for x in per_anchor if x > 0]
    return sum(per_anchor) / len(per_anchor) if per_anchor else 0.0


class TestPerAnchorReducer(unittest.TestCase):
    def test_matches_independent_per_anchor_mean(self):
        torch.manual_seed(0)
        batch_size = 40
        for aggregation in ("mean", "sum"):
            for inner in (MeanReducer(), AvgNonZeroReducer()):
                reducer = PerAnchorReducer(inner, aggregation=aggregation)
                for dtype in TEST_DTYPES:
                    embeddings = torch.randn(batch_size, 8, device=TEST_DEVICE)
                    embeddings = embeddings.to(dtype)
                    labels = torch.randint(0, 5, (batch_size,), device=TEST_DEVICE)
                    a1, p, a2, n = indexing.get_all_pairs_indices(labels)
                    triplets = indexing.get_all_triplets_indices(labels)
                    for indices in (
                        torch.arange(batch_size, device=TEST_DEVICE),
                        (a1, p),
                        (a2, n),
                        triplets,
                    ):
                        anchors = (
                            indices if isinstance(indices, torch.Tensor) else indices[0]
                        )
                        losses = torch.rand(len(anchors), device=TEST_DEVICE).to(dtype)
                        losses = torch.where(losses > 0.3, losses, 0)
                        output = reducer(
                            {"loss": {"losses": losses, "indices": indices}},
                            embeddings,
                            labels,
                        )
                        expected = per_anchor_reference(
                            losses, anchors, aggregation, inner
                        )
                        rtol = 1e-2 if dtype == torch.float16 else 1e-5
                        torch.testing.assert_close(
                            output.double().cpu(),
                            torch.tensor(expected, dtype=torch.float64),
                            rtol=rtol,
                            atol=1e-6,
                        )

    def test_references_larger_than_queries(self):
        embeddings = torch.randn(2, 3, device=TEST_DEVICE)
        labels = torch.tensor([0, 1], device=TEST_DEVICE)
        ref_emb = torch.randn(6, 3, device=TEST_DEVICE)
        ref_labels = torch.tensor([0, 0, 1, 1, 0, 1], device=TEST_DEVICE)
        loss = ContrastiveLoss(reducer=PerAnchorReducer())
        expected = ContrastiveLoss(reducer=MeanReducer()).loss_terms(
            embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels
        )
        reference = sum(
            per_anchor_reference(
                term["losses"], term["indices"][0], "mean", MeanReducer()
            )
            for term in expected.values()
        )
        actual = loss(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
        torch.testing.assert_close(
            actual.double().cpu(), torch.tensor(reference, dtype=torch.float64)
        )

    def test_duplicate_pairs_are_aggregated_with_gradients(self):
        embeddings = torch.zeros(2, 2, device=TEST_DEVICE)
        losses = torch.tensor([1.0, 1.0, 2.0], device=TEST_DEVICE, requires_grad=True)
        anchors = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        others = torch.tensor([1, 1, 0], device=TEST_DEVICE)
        output = PerAnchorReducer()(
            {"loss": {"losses": losses, "indices": (anchors, others)}},
            embeddings,
            None,
        )
        output.backward()
        torch.testing.assert_close(output, torch.tensor(1.5, device=TEST_DEVICE))
        torch.testing.assert_close(
            losses.grad, torch.tensor([0.25, 0.25, 0.5], device=TEST_DEVICE)
        )

    def test_rejects_unknown_aggregation(self):
        with self.assertRaises(ValueError):
            PerAnchorReducer(aggregation="max")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
