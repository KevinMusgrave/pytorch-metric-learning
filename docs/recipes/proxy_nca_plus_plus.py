import torch
from torch import Tensor, nn

from pytorch_metric_learning import losses


def global_k_max_pool(features: Tensor, k: int = 1) -> Tensor:
    flattened = features.flatten(start_dim=2)
    return torch.topk(flattened, k, dim=2).values.mean(dim=2)


class PoolAndEmbed(nn.Module):
    def __init__(self, channels: int, embedding_size: int, k: int = 1) -> None:
        super().__init__()
        self.k = k
        self.linear = nn.Linear(channels, embedding_size)
        self.norm = nn.LayerNorm(embedding_size, elementwise_affine=False)

    def forward(self, features: Tensor) -> Tensor:
        return self.norm(self.linear(global_k_max_pool(features, self.k)))


def make_optimizer(
    model: nn.Module, loss_func: nn.Module, lr: float, proxy_lr: float
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [
            {"params": model.parameters(), "lr": lr},
            {"params": loss_func.parameters(), "lr": proxy_lr},
        ]
    )


def train(
    model: nn.Module,
    loss_func: losses.ProxyNCALoss,
    optimizer: torch.optim.Optimizer,
    data: Tensor,
    labels: Tensor,
    steps: int = 30,
) -> list[float]:
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_func(model(data), labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def main() -> None:
    torch.manual_seed(17)
    labels = torch.arange(4).repeat_interleave(4)
    data = torch.randn(16, 6, 3, 3)
    model = PoolAndEmbed(channels=6, embedding_size=4)
    loss_func = losses.ProxyNCALoss(4, 4, softmax_scale=9)
    optimizer = make_optimizer(model, loss_func, lr=0.1, proxy_lr=1.0)
    history = train(model, loss_func, optimizer, data, labels)
    print(f"ProxyNCA++: {history[0]:.4f} -> {history[-1]:.4f}")


if __name__ == "__main__":
    main()
