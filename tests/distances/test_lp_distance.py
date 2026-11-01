import pytest
import torch

from pytorch_metric_learning.distances import LpDistance

from .. import TEST_DEVICE


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_half_precision_values_and_gradients(dtype, p, power, normalize):
    query = torch.tensor(
        [[0.25, -0.5, 1.0], [1.5, 0.75, -1.0]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    reference = torch.tensor(
        [[-1.0, 1.25, 0.5], [0.75, -1.5, 2.0], [-0.5, -0.25, -1.25]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    expected_query = query.detach().double().requires_grad_()
    expected_reference = reference.detach().double().requires_grad_()
    q, r = expected_query, expected_reference
    if normalize:
        q = q / torch.linalg.vector_norm(q, ord=p, dim=1, keepdim=True)
        r = r / torch.linalg.vector_norm(r, ord=p, dim=1, keepdim=True)
    expected = torch.linalg.vector_norm(q[:, None] - r[None, :], ord=p, dim=2)
    expected = expected**power

    distance = LpDistance(normalize_embeddings=normalize, p=p, power=power)
    actual = distance(query, reference)
    assert actual.dtype == (dtype if normalize else torch.float32)
    assert actual.device == query.device
    weights = query.new_tensor([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]])
    actual_gradients = torch.autograd.grad(
        (actual * weights).mean(), (query, reference)
    )
    expected_gradients = torch.autograd.grad(
        (expected * weights.double()).mean(), (expected_query, expected_reference)
    )
    tolerance = torch.finfo(dtype).eps
    for result, target in zip(
        (actual, *actual_gradients), (expected, *expected_gradients), strict=True
    ):
        torch.testing.assert_close(
            result.double(), target, rtol=4 * tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [24, 32])
def test_half_precision_nearby_rows(dtype, batch):
    query = torch.ones(batch, 128, dtype=dtype, device=TEST_DEVICE)
    reference = query.clone()
    reference[:, 0] = 1 - torch.finfo(dtype).eps
    query.requires_grad_()
    reference.requires_grad_()
    actual = LpDistance(normalize_embeddings=False)(query, reference)
    expected = torch.linalg.vector_norm(
        query.detach().double()[:, None] - reference.detach().double()[None, :], dim=2
    )
    gradient = torch.autograd.grad(actual.sum(), query)[0]
    tolerance = torch.finfo(dtype).eps
    assert torch.all(expected > 0)
    torch.testing.assert_close(actual.double(), expected, rtol=4 * tolerance, atol=0)
    torch.testing.assert_close(
        gradient.double()[:, 0], torch.full_like(expected[:, 0], float(batch))
    )
    assert torch.all(gradient[:, 1:] == 0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_pairwise_equals_matrix_diagonal(dtype, p, normalize):
    query = torch.tensor(
        [[0.25, -0.5, 1.0], [1.5, 0.75, -1.0], [0.5, 0.5, 0.5]],
        dtype=dtype,
        device=TEST_DEVICE,
    )
    reference = query.clone()
    reference[1] = torch.tensor([-1.0, 1.25, 0.5], dtype=dtype)
    distance = LpDistance(normalize_embeddings=normalize, p=p)
    pairwise = distance.pairwise(query, reference)
    assert pairwise.dtype == (
        dtype if normalize else torch.promote_types(dtype, torch.float32)
    )
    diagonal = distance(query, reference).diagonal()
    assert pairwise[0] == diagonal[0] == 0 and pairwise[2] == diagonal[2] == 0
    q, r = query.double(), reference.double()
    if normalize:
        q = q / torch.linalg.vector_norm(q, ord=p, dim=1, keepdim=True)
        r = r / torch.linalg.vector_norm(r, ord=p, dim=1, keepdim=True)
    expected = torch.linalg.vector_norm(q - r, ord=p, dim=1)
    tolerance = torch.finfo(dtype).eps
    for actual in (pairwise, diagonal):
        torch.testing.assert_close(
            actual.double(), expected, rtol=4 * tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("p", [1, 2, 3])
def test_pairwise_gradients_at_identical_rows_match_matrix_path(dtype, p):
    rows = torch.tensor(
        [[0.25, -0.5, 1.0], [1.5, 0.75, -1.0]], dtype=dtype, device=TEST_DEVICE
    )
    distance = LpDistance(normalize_embeddings=False, p=p)
    gradients = []
    for compute in (
        lambda x, y: distance.pairwise(x, y),
        lambda x, y: distance(x, y).diagonal(),
    ):
        query = rows.clone().requires_grad_()
        reference = rows.clone().requires_grad_()
        gradients.append(
            torch.autograd.grad(compute(query, reference).sum(), (query, reference))
        )
    for pairwise_gradient, matrix_gradient in zip(*gradients, strict=True):
        assert torch.isfinite(pairwise_gradient).all()
        assert torch.equal(pairwise_gradient, torch.zeros_like(pairwise_gradient))
        assert torch.equal(pairwise_gradient, matrix_gradient)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", ["random", "nearby", "mixed-scale", "duplicates"])
def test_large_low_precision_matrix_matches_direct_differences(dtype, normalize, kind):
    generator = torch.Generator().manual_seed(37)
    query = torch.randn(32, 128, generator=generator)
    reference = torch.randn(40, 128, generator=generator)
    if kind == "nearby":
        query.fill_(1)
        reference.fill_(1)
        reference[:, 0] = 1 - torch.finfo(dtype).eps
    elif kind == "mixed-scale":
        query.zero_()
        reference.zero_()
        query[:, 0] = reference[:, 0] = 1000
        reference[:, 1] = 2**-20
    elif kind == "duplicates":
        reference[:16] = query[:16]
    query = query.to(TEST_DEVICE, dtype).requires_grad_()
    reference = reference.to(TEST_DEVICE, dtype).requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    expected_reference = reference.detach().clone().requires_grad_()
    q, r = expected_query.float(), expected_reference.float()
    if normalize:
        q = (q / torch.linalg.vector_norm(q, dim=1, keepdim=True)).to(dtype).float()
        r = (r / torch.linalg.vector_norm(r, dim=1, keepdim=True)).to(dtype).float()
    expected = torch.linalg.vector_norm(q[:, None] - r[None, :], dim=-1)
    if normalize:
        expected = expected.to(dtype)
    actual = LpDistance(normalize_embeddings=normalize)(query, reference)
    weights = torch.randn(32, 40, generator=generator).to(TEST_DEVICE)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), (query, reference))
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), (expected_query, expected_reference)
    )
    for result, target in zip(
        (actual, *actual_gradients), (expected, *expected_gradients), strict=True
    ):
        torch.testing.assert_close(
            result, target, rtol=4 * torch.finfo(dtype).eps, atol=torch.finfo(dtype).eps
        )
    if kind in ("nearby", "mixed-scale") and not normalize:
        assert (actual > 0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mode", ["eager", "compiled"] + (["captured"] if TEST_DEVICE.type == "cuda" else [])
)
def test_low_precision_self_matrix_preserves_graph_execution(dtype, mode):
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(32, 16, generator=generator).to(TEST_DEVICE, dtype)
    query.requires_grad_()
    reference = query.detach().float().requires_grad_()
    expected = torch.linalg.vector_norm(reference[:, None] - reference[None, :], dim=-1)
    expected_gradient = torch.autograd.grad(expected.sum(), reference)[0].to(dtype)
    distance = LpDistance(normalize_embeddings=False)
    if mode == "compiled":
        distance = torch.compile(distance, backend="eager", fullgraph=True)
    if mode == "captured":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                query.grad = None
                distance(query).sum().backward()
        torch.cuda.current_stream().wait_stream(stream)
        query.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = distance(query)
            actual.sum().backward()
        graph.replay()
    else:
        actual = distance(query)
        actual.sum().backward()
    assert torch.count_nonzero(actual.diagonal()) == 0
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        query.grad,
        expected_gradient,
        rtol=4 * torch.finfo(dtype).eps,
        atol=torch.finfo(dtype).eps,
    )


@pytest.mark.parametrize("precision", ["high", "medium"])
def test_duplicate_rows_remain_zero_with_reduced_matmul_precision(precision):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        rows = (
            torch.tensor(
                [[-0.87109375, 0.91796875]], dtype=torch.bfloat16, device=TEST_DEVICE
            )
            .repeat(32, 1)
            .requires_grad_()
        )
        matrix = LpDistance()(rows)
        gradient = torch.autograd.grad(matrix.sum(), rows)[0]
        assert torch.count_nonzero(matrix) == 0
        assert torch.count_nonzero(gradient) == 0
    finally:
        torch.set_float32_matmul_precision(previous)
