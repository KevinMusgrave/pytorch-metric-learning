import re
import unittest
from pathlib import Path

import torch

from pytorch_metric_learning import losses

DOCS = Path(__file__).resolve().parents[2] / "docs"
LEARNABLE = {
    "ArcFaceLoss": (3, 4),
    "CosFaceLoss": (3, 4),
    "LargeMarginSoftmaxLoss": (3, 4),
    "NormalizedSoftmaxLoss": (3, 4),
    "P2SGradLoss": (3, 4),
    "ProxyAnchorLoss": (3, 4),
    "ProxyNCALoss": (3, 4),
    "SoftTripleLoss": (3, 4),
    "SphereFaceLoss": (3, 4),
    "SubCenterArcFaceLoss": (3, 4),
}
CONSTRUCTOR_ARGUMENTS = {"RankedListLoss": (0.5, 1.0)}
NOT_PROBED = {
    "BaseMetricLossFunction",
    "BaseProxyLoss",
    "ManifoldLoss",
    "CrossBatchMemory",
    "GenericPairLoss",
    "MultipleLosses",
    "SelfSupervisedLoss",
    "VICRegLoss",
}


def documented_list(heading: str) -> set[str]:
    text = (DOCS / "losses.md").read_text()
    section = text.split(heading, 1)[1].split("\n\n", 2)[1]
    return set(re.findall(r"^- (\w+)$", section, re.MULTILINE))


def construct(name: str) -> losses.BaseMetricLossFunction:
    cls = getattr(losses, name)
    return cls(*LEARNABLE.get(name, CONSTRUCTOR_ARGUMENTS.get(name, ())))


def labels_for(loss: losses.BaseMetricLossFunction, labels: torch.Tensor):
    if loss.label_ndim == 1:
        return labels
    return torch.nn.functional.one_hot(labels, num_classes=3).bool()


def accepts(loss: torch.nn.Module, **kwargs) -> bool:
    try:
        loss(**kwargs)
    except ValueError:
        return False
    return True


class TestDocumentedCapabilities(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.embeddings = torch.randn(6, 4)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.ref_emb = torch.randn(5, 4)
        self.ref_labels = torch.tensor([0, 1, 2, 0, 1])
        self.pairs = (
            torch.tensor([0, 2]),
            torch.tensor([1, 3]),
            torch.tensor([0, 2]),
            torch.tensor([2, 4]),
        )

    def probed_losses(self) -> list[str]:
        return [name for name in losses.__all__ if name not in NOT_PROBED]

    def test_indices_without_labels_list_matches_behavior(self) -> None:
        documented = documented_list(
            "**Losses that accept `indices_tuple` without `labels`**"
        )
        observed = set()
        for name in self.probed_losses():
            with self.subTest(loss=name):
                loss = construct(name)
                if accepts(loss, embeddings=self.embeddings, indices_tuple=self.pairs):
                    observed.add(name)
        self.assertEqual(observed, documented)

    def test_reference_embeddings_list_matches_behavior(self) -> None:
        documented = documented_list(
            "**Losses that accept `ref_emb` and `ref_labels`**"
        )
        observed = set()
        for name in self.probed_losses():
            with self.subTest(loss=name):
                loss = construct(name)
                if accepts(
                    loss,
                    embeddings=self.embeddings,
                    labels=labels_for(loss, self.labels),
                    ref_emb=self.ref_emb,
                    ref_labels=labels_for(loss, self.ref_labels),
                ):
                    observed.add(name)
        self.assertEqual(observed, documented)

    def test_every_loss_is_documented(self) -> None:
        pages = {path.stem for path in (DOCS / "losses").glob("*.md")}
        overview = (DOCS / "losses.md").read_text()
        for name in losses.__all__:
            with self.subTest(loss=name):
                self.assertIn(f"[{name}](losses/", overview)
        self.assertEqual(len(pages), len(losses.__all__))
