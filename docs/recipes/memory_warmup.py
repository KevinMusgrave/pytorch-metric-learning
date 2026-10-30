import torch
from torch import Tensor, nn

from pytorch_metric_learning import losses


def train_with_memory(
    model: nn.Module,
    memory: losses.CrossBatchMemory,
    data: Tensor,
    labels: Tensor,
    warmup_steps: int = 2,
    steps: int = 6,
) -> list[float]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(data)
        if step < warmup_steps:
            loss = memory.loss(embeddings, labels)
        else:
            loss = memory(embeddings, labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def main() -> None:
    torch.manual_seed(17)
    model = nn.Linear(8, 4)
    data = torch.randn(12, 8)
    labels = torch.arange(4).repeat_interleave(3)
    memory = losses.CrossBatchMemory(
        losses.ContrastiveLoss(neg_margin=1.4), embedding_size=4, memory_size=32
    )
    history = train_with_memory(model, memory, data, labels)
    print(
        f"loss: {history[0]:.4f} -> {history[-1]:.4f}; queue cursor: {memory.queue_idx}"
    )


if __name__ == "__main__":
    main()
