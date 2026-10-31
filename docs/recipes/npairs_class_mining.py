import torch
from torch import Tensor, nn

from pytorch_metric_learning import distances, losses


def candidate_examples(
    labels: Tensor,
    num_classes: int,
    examples_per_class: int = 2,
    generator: torch.Generator | None = None,
) -> Tensor:
    classes, counts = torch.unique(labels, return_counts=True)
    eligible = classes[counts >= 2]
    if len(eligible) < num_classes:
        raise ValueError("num_classes exceeds the classes with at least two examples")
    order = torch.randperm(len(eligible), generator=generator)
    selected = []
    for label in eligible[order[:num_classes].to(eligible.device)]:
        rows = torch.where(labels == label)[0]
        draw = torch.randperm(len(rows), generator=generator)[:examples_per_class]
        selected.append(rows[draw.to(rows.device)])
    return torch.cat(selected)


def class_violations(
    embeddings: Tensor, labels: Tensor, distance: distances.BaseDistance
) -> tuple[Tensor, Tensor]:
    similarity = distance(embeddings)
    if not distance.is_inverted:
        similarity = -similarity
    classes, class_index = torch.unique(labels, return_inverse=True)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same.fill_diagonal_(False)
    hardest = similarity.masked_fill(~same, torch.inf).amin(dim=1, keepdim=True)
    hardest = torch.where(same.any(dim=1, keepdim=True), hardest, 0)
    excess = similarity - hardest
    rows, size = len(labels), len(classes)
    per_class = excess.new_full((rows, size), -torch.inf).scatter_reduce(
        1, class_index.expand(rows, -1), excess, reduce="amax"
    )
    violations = excess.new_full((size, size), -torch.inf).scatter_reduce(
        0, class_index.unsqueeze(1).expand(-1, size), per_class, reduce="amax"
    )
    return classes, violations


def select_classes(
    embeddings: Tensor,
    labels: Tensor,
    num_pairs: int,
    distance: distances.BaseDistance,
    generator: torch.Generator | None = None,
) -> Tensor:
    classes, violations = class_violations(embeddings, labels, distance)
    if not 0 < num_pairs <= len(classes):
        raise ValueError("num_pairs must be between 1 and the candidate classes")
    first = int(torch.randint(len(classes), (1,), generator=generator))
    selected = [first]
    score = violations[first].clone()
    while len(selected) < num_pairs:
        score[selected] = -torch.inf
        tied = torch.where(score == score.max())[0]
        choice = int(torch.randint(len(tied), (1,), generator=generator))
        selected.append(int(tied[choice]))
        score = torch.maximum(score, violations[selected[-1]])
    return classes[selected]


def draw_pairs(
    labels: Tensor, classes: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    pairs = []
    for label in classes:
        rows = torch.where(labels == label)[0]
        if len(rows) < 2:
            raise ValueError("Every selected class needs two examples")
        draw = torch.randperm(len(rows), generator=generator)[:2]
        pairs.append(rows[draw.to(rows.device)])
    return torch.cat(pairs)


def mined_batch(
    model: nn.Module,
    data: Tensor,
    labels: Tensor,
    num_candidates: int,
    num_pairs: int,
    distance: distances.BaseDistance,
    generator: torch.Generator | None = None,
) -> Tensor:
    rows = candidate_examples(labels, num_candidates, generator=generator)
    with torch.no_grad():
        embeddings = model(data[rows])
    classes = select_classes(embeddings, labels[rows], num_pairs, distance, generator)
    return draw_pairs(labels, classes, generator)


def main() -> None:
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(16).repeat_interleave(4)
    data = torch.randn(16, 8)[labels] + 0.5 * torch.randn(64, 8)
    model = nn.Linear(8, 4)
    loss_func = losses.NPairsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    history = []
    for _ in range(40):
        batch = mined_batch(model, data, labels, 12, 4, loss_func.distance, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_func(model(data[batch]), labels[batch])
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    print(f"mined N-pairs: {history[0]:.4f} -> {history[-1]:.4f}")


if __name__ == "__main__":
    main()
