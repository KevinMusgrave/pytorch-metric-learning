import torch
from torch import Tensor, nn

from pytorch_metric_learning import losses, miners


def train_metric(
    model: nn.Module,
    data: Tensor,
    labels: Tensor,
    classifier: nn.Module | None = None,
    steps: int = 30,
) -> list[float]:
    parameters = list(model.parameters())
    if classifier is not None:
        parameters.extend(classifier.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    loss_func = losses.ContrastiveLoss(neg_margin=1.4)
    miner = miners.PairMarginMiner(pos_margin=0, neg_margin=1.4)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(data)
        pairs = miner(embeddings, labels)
        loss = loss_func(embeddings, labels, pairs)
        if classifier is not None:
            loss = loss + nn.functional.cross_entropy(classifier(embeddings), labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def train_proxy(
    model: nn.Module,
    loss_func: losses.NormalizedSoftmaxLoss,
    data: Tensor,
    labels: Tensor,
    steps: int = 30,
) -> list[float]:
    optimizer = torch.optim.SGD([*model.parameters(), *loss_func.parameters()], lr=0.1)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_func(model(data), labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def train_two_stream(
    query_model: nn.Module,
    reference_model: nn.Module,
    queries: Tensor,
    query_labels: Tensor,
    references: Tensor,
    reference_labels: Tensor,
    steps: int = 30,
) -> list[float]:
    optimizer = torch.optim.SGD(
        [*query_model.parameters(), *reference_model.parameters()], lr=0.1
    )
    loss_func = losses.ContrastiveLoss(neg_margin=1.4)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_func(
            query_model(queries),
            query_labels,
            ref_emb=reference_model(references),
            ref_labels=reference_labels,
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def main() -> None:
    torch.manual_seed(17)
    data = torch.randn(12, 8)
    labels = torch.arange(4).repeat_interleave(3)
    histories = {
        "metric": train_metric(nn.Linear(8, 4), data, labels),
        "classifier": train_metric(
            nn.Linear(8, 4), data, labels, classifier=nn.Linear(4, 4)
        ),
        "proxy": train_proxy(
            nn.Linear(8, 4),
            losses.NormalizedSoftmaxLoss(4, 4, temperature=0.1),
            data,
            labels,
        ),
        "two_stream": train_two_stream(
            nn.Linear(8, 4), nn.Linear(8, 4), data, labels, data[:8], labels[:8]
        ),
    }
    for name, history in histories.items():
        print(f"{name}: {history[0]:.4f} -> {history[-1]:.4f}")


if __name__ == "__main__":
    main()
