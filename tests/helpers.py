import math
from collections.abc import Callable, Sequence

import torch

from . import TEST_DEVICE

LOW_PRECISION = (torch.float16, torch.bfloat16)


def use_deterministic_kernels(test) -> None:
    torch.use_deterministic_algorithms(True)
    test.addCleanup(torch.use_deterministic_algorithms, False)


def angle_to_coord(angle, normalized=False):
    radians = math.radians(angle)
    x = math.cos(radians)
    y = math.sin(radians)
    if not normalized:
        x *= 2
        y *= 0.5
    return x, y


def planar_embeddings(angles, dtype, normalized=False, requires_grad=True):
    return torch.tensor(
        [angle_to_coord(a, normalized=normalized) for a in angles],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=requires_grad,
    )


def rtol(dtype: torch.dtype) -> float:
    return 1e-2 if dtype in LOW_PRECISION else 1e-5


def assert_close(actual, expected, dtype: torch.dtype, atol: float = 1e-6) -> None:
    low = dtype in LOW_PRECISION
    torch.testing.assert_close(
        torch.as_tensor(actual).double(),
        torch.as_tensor(expected).double().to(torch.as_tensor(actual).device),
        rtol=rtol(dtype),
        atol=max(atol, 1e-2 if low else 0),
    )


def assert_matches_float64(
    compute: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    inputs: Sequence[torch.Tensor],
    dtype: torch.dtype,
    atol: float = 1e-6,
) -> torch.Tensor:
    leaves = [x.detach().to(TEST_DEVICE, dtype).requires_grad_() for x in inputs]
    exact = [x.detach().double().requires_grad_() for x in leaves]
    actual = compute(*leaves)
    expected = reference(*exact)
    assert_close(actual.detach(), expected.detach(), dtype, atol)
    actual.backward()
    expected.backward()
    for leaf, exact_leaf in zip(leaves, exact, strict=True):
        assert leaf.grad is not None and exact_leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()
        assert_close(leaf.grad, exact_leaf.grad, dtype, atol)
    return actual
