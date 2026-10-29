import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import indexing


def generator_losses(
    embedder: nn.Module,
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    synthetic: Tensor,
    margin: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    a, p, n = (F.normalize(embedder(x), dim=1) for x in (anchors, positives, synthetic))
    violations = F.relu(
        torch.linalg.vector_norm(a - n, dim=1)
        - torch.linalg.vector_norm(a - p, dim=1)
        - margin
    )
    adversarial = violations.sum() / (violations > 0).sum().clamp_min(1)
    hardness = F.mse_loss(F.normalize(synthetic, dim=1), F.normalize(anchors, dim=1))
    regularization = F.mse_loss(
        F.normalize(synthetic, dim=1), F.normalize(negatives, dim=1)
    )
    return adversarial, hardness, regularization


def generator_step(
    trunk: nn.Module,
    embedder: nn.Module,
    generator: nn.Module,
    data: Tensor,
    triplets: tuple[Tensor, Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        features = trunk(data)
    a, p, n = (features[index] for index in triplets)
    synthetic = generator(torch.cat((a, p, n), dim=1))
    parameter_flags = [parameter.requires_grad for parameter in embedder.parameters()]
    embedder.requires_grad_(False)
    try:
        adv, hard, reg = generator_losses(embedder, a, p, n, synthetic)
        loss = adv + hard + reg
        loss.backward()
        optimizer.step()
    finally:
        for parameter, flag in zip(embedder.parameters(), parameter_flags, strict=True):
            parameter.requires_grad_(flag)
    return loss.detach()


def metric_step(
    trunk: nn.Module,
    embedder: nn.Module,
    generator: nn.Module,
    data: Tensor,
    labels: Tensor,
    triplets: tuple[Tensor, Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    features = trunk(data)
    a, p, n = (features[index] for index in triplets)
    with torch.no_grad():
        synthetic = generator(torch.cat((a, p, n), dim=1))
    packaged = torch.stack((a, p, synthetic), dim=1).flatten(0, 1)
    anchors = torch.arange(0, len(packaged), 3, device=data.device)
    loss_func = losses.TripletMarginLoss(margin=0.2)
    authentic_loss = loss_func(embedder(features), labels, triplets)
    synthetic_loss = loss_func(
        embedder(packaged), indices_tuple=(anchors, anchors + 1, anchors + 2)
    )
    loss = authentic_loss + synthetic_loss
    loss.backward()
    optimizer.step()
    return loss.detach()


def main() -> None:
    torch.manual_seed(17)
    trunk = nn.Linear(8, 6)
    embedder = nn.Linear(6, 4)
    generator = nn.Sequential(nn.Linear(18, 12), nn.ReLU(), nn.Linear(12, 6))
    model_optimizer = torch.optim.SGD(
        [*trunk.parameters(), *embedder.parameters()], lr=0.05
    )
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.05)
    data = torch.randn(12, 8)
    labels = torch.arange(4).repeat_interleave(3)
    triplets = indexing.get_all_triplets_indices(labels)
    for _ in range(10):
        generator_loss = generator_step(
            trunk, embedder, generator, data, triplets, generator_optimizer
        )
        metric_loss = metric_step(
            trunk, embedder, generator, data, labels, triplets, model_optimizer
        )
    print(f"generator: {float(generator_loss):.4f}; metric: {float(metric_loss):.4f}")


if __name__ == "__main__":
    main()
