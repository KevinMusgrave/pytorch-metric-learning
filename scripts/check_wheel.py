import tomllib
from importlib.metadata import requires, version
from pathlib import Path

import torch

import pytorch_metric_learning as pml
from pytorch_metric_learning.losses import ContrastiveLoss


def main():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    installed = version("pytorch-metric-learning")
    if installed != project["version"]:
        raise RuntimeError("Installed wheel and source metadata versions must match")
    requirements = requires("pytorch-metric-learning")
    if requirements is None:
        raise RuntimeError("Installed wheel must declare its runtime requirements")
    for name in ("numpy", "torch"):
        minimum = f"{name}>={version(name).split('+')[0]}"
        if minimum not in requirements:
            raise RuntimeError(f"Wheel check requires the minimum runtime: {minimum}")
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0], [-1.0, 0.1]], requires_grad=True
    )
    loss = ContrastiveLoss()(embeddings, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    if embeddings.grad is None:
        raise RuntimeError("Installed wheel must propagate gradients to embeddings")
    if not torch.isfinite(loss) or not torch.isfinite(embeddings.grad).all():
        raise RuntimeError("Installed wheel must produce finite loss and gradients")
    if not Path(pml.__file__).with_name("py.typed").is_file():
        raise RuntimeError("Installed wheel must include py.typed")
    print(f"Installed wheel {installed} passed with minimum supported runtimes")


if __name__ == "__main__":
    main()
