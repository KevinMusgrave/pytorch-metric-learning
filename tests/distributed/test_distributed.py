import unittest
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.distributed import (
    DistributedLossWrapper,
    DistributedMinerWrapper,
)
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    CrossBatchMemory,
    NormalizedSoftmaxLoss,
)
from pytorch_metric_learning.miners import BatchEasyHardMiner, PairMarginMiner

from .. import TEST_DEVICE, TEST_DTYPES


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    else:
        torch.testing.assert_close(actual, expected)


def contrastive_reference(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
    self_indices: torch.Tensor | None = None,
    pairs: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(query[:, None] - reference[None, :], dim=-1)
    positive = query_labels[:, None] == reference_labels[None, :]
    negative = ~positive
    if self_indices is not None:
        positive[torch.arange(len(query), device=query.device), self_indices] = False
    if pairs is not None:
        positive = torch.zeros_like(positive)
        negative = torch.zeros_like(negative)
        positive[pairs[0], pairs[1]] = True
        negative[pairs[2], pairs[3]] = True
    positives = F.relu(distances[positive] - 0.1)
    negatives = F.relu(1.2 - distances[negative])
    return positives.sum() / positives.count_nonzero().clamp_min(
        1
    ) + negatives.sum() / negatives.count_nonzero().clamp_min(1)


def observed_gradient(tensor: torch.Tensor) -> torch.Tensor:
    gradient = tensor.grad
    assert gradient is not None
    return gradient


def make_pair_loss() -> ContrastiveLoss:
    return ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=1.2,
        distance=LpDistance(normalize_embeddings=False),
    )


def make_inputs(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.8, 0.5], [0.7, 0.1], [0.4, -0.2]],
        device=device,
        dtype=dtype,
    )
    return embeddings, torch.tensor([0, 0, 1, 1, 0], device=device)


def rank_slice(sizes: tuple[int, int], rank: int) -> slice:
    start = sum(sizes[:rank])
    return slice(start, start + sizes[rank])


def check_global_pairs(rank: int, device: torch.device, dtype: torch.dtype) -> None:
    inputs, labels = make_inputs(device, dtype)
    for sizes in ((2, 3), (0, 5), (5, 0), (0, 0)):
        selected = rank_slice(sizes, rank)
        query = inputs[: sum(sizes)].clone().requires_grad_()
        query_labels = labels[: sum(sizes)]
        self_indices = torch.arange(len(query), device=device)
        expected = contrastive_reference(
            query, query_labels, query, query_labels, self_indices
        )
        expected_gradient = torch.autograd.grad(expected, query)[0]
        local = query.detach()[selected].clone().requires_grad_()
        actual = DistributedLossWrapper(make_pair_loss())(local, query_labels[selected])
        actual.backward()
        assert_close(actual, expected)
        assert_close(observed_gradient(local) / 2, expected_gradient[selected])


def check_reference_pairs(rank: int, device: torch.device, dtype: torch.dtype) -> None:
    inputs, labels = make_inputs(device, dtype)
    query = inputs.clone().requires_grad_()
    reference = (inputs[:4] + 0.15).requires_grad_()
    reference_labels = torch.tensor([0, 1, 0, 1], device=device)
    query_slice, reference_slice = rank_slice((2, 3), rank), rank_slice((3, 1), rank)
    local = query.detach()[query_slice].clone().requires_grad_()
    local_reference = reference.detach()[reference_slice].clone().requires_grad_()
    miner = DistributedMinerWrapper(
        PairMarginMiner(
            pos_margin=0.2,
            neg_margin=0.9,
            distance=LpDistance(normalize_embeddings=False),
        )
    )
    pairs = miner(
        local,
        labels[query_slice],
        ref_emb=local_reference,
        ref_labels=reference_labels[reference_slice],
    )
    distances = torch.linalg.vector_norm(query[:, None] - reference[None, :], dim=-1)
    matches = labels[:, None] == reference_labels[None, :]
    expected_pairs = (
        *torch.where(matches & (distances > 0.2)),
        *torch.where(~matches & (distances < 0.9)),
    )
    for actual_indices, expected_indices in zip(pairs, expected_pairs, strict=True):
        assert_close(actual_indices, expected_indices)
    expected = contrastive_reference(
        query, labels, reference, reference_labels, pairs=expected_pairs
    )
    query_gradient, reference_gradient = torch.autograd.grad(
        expected, (query, reference)
    )
    actual = DistributedLossWrapper(make_pair_loss())(
        local, indices_tuple=pairs, ref_emb=local_reference
    )
    actual.backward()
    assert_close(actual, expected)
    assert_close(observed_gradient(local) / 2, query_gradient[query_slice])
    assert_close(
        observed_gradient(local_reference) / 2, reference_gradient[reference_slice]
    )


class ProxyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.proxy_loss = NormalizedSoftmaxLoss(
            num_classes=2, embedding_size=2, temperature=0.5
        )
        self.objective = DistributedLossWrapper(self.proxy_loss)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.objective(self.encoder(inputs), labels)


def check_trainable_loss(rank: int, device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    model = ProxyModel().to(device=device, dtype=dtype)
    ddp = DistributedDataParallel(
        model, device_ids=[rank] if device.type == "cuda" else None
    )
    inputs, labels = make_inputs(device, dtype)
    encoder = model.encoder.weight.detach().clone().requires_grad_()
    proxies = model.proxy_loss.weight.detach().clone().requires_grad_()
    embedded = F.normalize(F.linear(inputs, encoder), dim=1)
    logits = embedded @ F.normalize(proxies, dim=1).t() / 0.5
    expected = -F.log_softmax(logits, dim=1)[
        torch.arange(len(labels), device=device), labels
    ].mean()
    encoder_gradient, proxy_gradient = torch.autograd.grad(expected, (encoder, proxies))
    selected = rank_slice((2, 3), rank)
    actual = ddp(inputs[selected], labels[selected])
    actual.backward()
    assert_close(actual, expected)
    assert_close(observed_gradient(model.encoder.weight), encoder_gradient)
    assert_close(observed_gradient(model.proxy_loss.weight), proxy_gradient)


def check_memory(rank: int, device: torch.device, dtype: torch.dtype) -> None:
    inputs, labels = make_inputs(device, dtype)
    selected = rank_slice((2, 3), rank)
    for mask in (None, torch.tensor([True, False, True, False, False], device=device)):
        reference = CrossBatchMemory(
            make_pair_loss(), embedding_size=2, memory_size=7
        ).to(TEST_DEVICE, dtype)
        memory = CrossBatchMemory(make_pair_loss(), embedding_size=2, memory_size=7)
        memory = memory.to(TEST_DEVICE, dtype)
        wrapped = DistributedLossWrapper(memory)
        for step in range(4):
            query = (inputs + step * 0.2).requires_grad_()
            expected = reference(query, labels, enqueue_mask=mask)
            expected_gradient = torch.autograd.grad(expected, query)[0]
            local = query.detach()[selected].clone().requires_grad_()
            local_mask = mask[selected] if mask is not None else None
            actual = wrapped(local, labels[selected], enqueue_mask=local_mask)
            actual.backward()
            assert_close(actual, expected)
            assert_close(observed_gradient(local) / 2, expected_gradient[selected])
            assert_close(memory.embedding_memory, reference.embedding_memory)
            assert_close(memory.label_memory, reference.label_memory)
            assert memory.queue_idx == reference.queue_idx
            assert memory.has_been_filled == reference.has_been_filled


def check_local_anchor_case(
    rank: int, device: torch.device, dtype: torch.dtype, sizes: tuple[int, int]
) -> None:
    inputs, labels = make_inputs(device, dtype)
    selected = rank_slice(sizes, rank)
    for use_reference in (False, True):
        query = inputs[selected].clone().requires_grad_()
        references = (
            (inputs + 0.15).requires_grad_() if use_reference else inputs.clone()
        )
        if not use_reference:
            references = torch.cat(
                (references[: selected.start], query, references[selected.stop :])
            )
        self_indices = (
            None
            if use_reference
            else torch.arange(selected.start, selected.stop, device=device)
        )
        expected = (
            contrastive_reference(
                query, labels[selected], references, labels, self_indices
            )
            * 2
        )
        expected_gradient = torch.autograd.grad(expected, query)[0]
        local = query.detach().clone().requires_grad_()
        ref_emb = references.detach()[selected] if use_reference else None
        actual = DistributedLossWrapper(make_pair_loss(), objective="local_anchor")(
            local,
            labels[selected],
            ref_emb=ref_emb,
            ref_labels=labels[selected] if use_reference else None,
        )
        actual.backward()
        assert_close(actual, expected)
        assert_close(observed_gradient(local), expected_gradient)


def check_local_miner(
    rank: int, device: torch.device, dtype: torch.dtype, sizes: tuple[int, int]
) -> None:
    inputs, labels = make_inputs(device, dtype)
    selected = rank_slice(sizes, rank)
    miner = DistributedMinerWrapper(
        PairMarginMiner(
            pos_margin=0.2,
            neg_margin=0.9,
            distance=LpDistance(normalize_embeddings=False),
        ),
        objective="local_anchor",
    )
    for use_reference in (False, True):
        references = inputs + 0.15 if use_reference else inputs
        local = inputs[selected]
        actual = miner(
            local,
            labels[selected],
            ref_emb=references[selected] if use_reference else None,
            ref_labels=labels[selected] if use_reference else None,
        )
        distances = torch.linalg.vector_norm(
            local[:, None] - references[None, :], dim=-1
        )
        positive = labels[selected, None] == labels[None, :]
        negative = ~positive
        if not use_reference:
            positive[
                torch.arange(len(local), device=device),
                torch.arange(selected.start, selected.stop, device=device),
            ] = False
        expected = (
            *torch.where(positive & (distances > 0.2)),
            *torch.where(negative & (distances < 0.9)),
        )
        for actual_indices, expected_indices in zip(actual, expected, strict=True):
            assert_close(actual_indices, expected_indices)


def check_local_selecting_miner(
    rank: int, device: torch.device, dtype: torch.dtype, sizes: tuple[int, int]
) -> None:
    inputs, labels = make_inputs(device, dtype)
    selected = rank_slice(sizes, rank)
    miner = DistributedMinerWrapper(
        BatchEasyHardMiner(
            pos_strategy="easy",
            neg_strategy="hard",
            distance=LpDistance(normalize_embeddings=False),
        ),
        objective="local_anchor",
    )
    actual = miner(inputs[selected], labels[selected])
    distances = torch.linalg.vector_norm(
        inputs[selected, None] - inputs[None, :], dim=-1
    )
    local = torch.arange(len(distances), device=device)
    positive = labels[selected, None] == labels[None, :]
    positive[local, local + selected.start] = False
    negative = labels[selected, None] != labels[None, :]
    expected = (
        local,
        distances.masked_fill(~positive, torch.inf).argmin(dim=1),
        local,
        distances.masked_fill(~negative, torch.inf).argmin(dim=1),
    )
    for actual_indices, expected_indices in zip(actual, expected, strict=True):
        assert_close(actual_indices, expected_indices)


def assert_raises_on_all_ranks(
    rank: int,
    local_message: str,
    call: Callable[[], object],
    local_error: type[Exception] = ValueError,
) -> None:
    error, message = (
        (local_error, local_message)
        if rank == 1
        else (ValueError, r"ranks \[1\] are invalid")
    )
    with unittest.TestCase().assertRaisesRegex(error, message):
        call()


def check_inconsistent_arguments(rank: int, device: torch.device) -> None:
    inputs, labels = make_inputs(device, torch.float32)
    local = inputs[:2].clone().requires_grad_()
    wrapper = DistributedLossWrapper(make_pair_loss())
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "same wrapper objective and optional arguments"
    ):
        wrapper(local, labels[:2], ref_emb=local.detach() if rank else None)
    malformed_labels = labels[:1] if rank else labels[:2]
    assert_raises_on_all_ranks(
        rank,
        "Number of embeddings must equal number of labels",
        lambda: wrapper(local, malformed_labels),
    )
    misplaced_labels = labels[:2].to("meta") if rank else labels[:2]
    assert_raises_on_all_ranks(
        rank, "labels are on meta", lambda: wrapper(local, misplaced_labels)
    )
    reference_labels = labels[:2].to("meta") if rank else labels[:2]
    assert_raises_on_all_ranks(
        rank,
        "ref_labels are on meta",
        lambda: wrapper(
            local, labels[:2], ref_emb=local.detach(), ref_labels=reference_labels
        ),
    )
    index = torch.arange(2, device=device)
    pairs = (index, index, index, index.float() if rank else index)
    assert_raises_on_all_ranks(
        rank,
        "int64 or int32",
        lambda: wrapper(local, labels[:2], indices_tuple=pairs),
        TypeError,
    )
    miner = DistributedMinerWrapper(PairMarginMiner())
    assert_raises_on_all_ranks(
        rank, "labels are on meta", lambda: miner(local, misplaced_labels)
    )
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "not supported across ranks"
    ):
        miner(local, labels[:2], indices_tuple=(index, index, index, index))
    mask = torch.ones(3 if rank else 2, device=device, dtype=torch.bool)
    memory = DistributedLossWrapper(
        CrossBatchMemory(make_pair_loss(), embedding_size=2).to(TEST_DEVICE)
    )
    assert_raises_on_all_ranks(
        rank,
        "boolean vector",
        lambda: memory(local, labels[:2], enqueue_mask=mask),
    )
    assert memory.loss.queue_idx == 0
    with unittest.TestCase().assertRaisesRegex(ValueError, "boolean vector"):
        memory(local, labels[:2], enqueue_mask=torch.ones(2, device=device))
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "matching dimensions, dtypes"
    ):
        wrapper(local.double() if rank else local, labels[:2])


def run_validation(rank: int, address: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=address,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        check_inconsistent_arguments(rank, torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def run_references(rank: int, address: str, backend: str) -> None:
    device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=address,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        for dtype in TEST_DTYPES:
            check_global_pairs(rank, device, dtype)
            check_reference_pairs(rank, device, dtype)
            check_trainable_loss(rank, device, dtype)
            check_memory(rank, device, dtype)
            for sizes in ((2, 3), (0, 5), (5, 0)):
                check_local_anchor_case(rank, device, dtype, sizes)
                check_local_miner(rank, device, dtype, sizes)
                check_local_selecting_miner(rank, device, dtype, sizes)
    finally:
        dist.destroy_process_group()


class TestDistributedReferences(unittest.TestCase):
    def test_two_rank_references(self) -> None:
        backend = "nccl" if TEST_DEVICE.type == "cuda" else "gloo"
        if backend == "nccl":
            self.assertGreaterEqual(
                torch.cuda.device_count(),
                2,
                "Distributed CUDA tests require two visible GPUs",
            )
        with TemporaryDirectory() as directory:
            address = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(run_references, args=(address, backend), nprocs=2, join=True)

    def test_two_rank_validation_raises_on_every_rank(self) -> None:
        with TemporaryDirectory() as directory:
            address = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(run_validation, args=(address,), nprocs=2, join=True)
