import torch
from torch import Tensor, nn

from pytorch_metric_learning import losses, miners


def cascaded_loss(
    embeddings: Tensor,
    labels: Tensor,
    sizes: list[int],
    stage_miners: tuple[miners.HDCMiner, ...],
    loss_func: nn.Module,
) -> tuple[Tensor, list[tuple[Tensor, ...]]]:
    stages = torch.split(embeddings, sizes, dim=1)
    previous = None
    selected = []
    total = embeddings.sum() * 0
    for stage, miner in zip(stages, stage_miners, strict=True):
        previous = miner(stage, labels, indices_tuple=previous)
        selected.append(previous)
        total = total + loss_func(stage, labels, previous)
    return total, selected


def main() -> None:
    torch.manual_seed(17)
    model = nn.Linear(8, 12)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.randn(12, 8)
    labels = torch.arange(4).repeat_interleave(3)
    stage_miners = tuple(miners.HDCMiner(p) for p in (1.0, 0.5, 0.5))
    loss_func = losses.ContrastiveLoss(neg_margin=1.4)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss, selected = cascaded_loss(
            model(data), labels, [4, 4, 4], stage_miners, loss_func
        )
        loss.backward()
        optimizer.step()
    print(f"loss: {float(loss.detach()):.4f}")
    print(f"positive pairs per stage: {[len(pairs[0]) for pairs in selected]}")


if __name__ == "__main__":
    main()
