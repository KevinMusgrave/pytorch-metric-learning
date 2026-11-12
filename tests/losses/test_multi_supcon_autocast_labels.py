import pytest
import torch

from pytorch_metric_learning.losses import CrossBatchMemory, MultiSupConLoss

from .test_multi_supcon_loss import reference_loss


def label_sets(cardinality):
    labels = torch.zeros(4, 2 * cardinality, dtype=torch.bool)
    labels[:2, :cardinality] = True
    labels[2:, cardinality:] = True
    return labels


@pytest.mark.parametrize(
    "dtype,cardinality", [(torch.bfloat16, 257), (torch.float16, 2049)]
)
@pytest.mark.parametrize("separate", [False, True])
def test_autocast_preserves_label_set_pairs(dtype, cardinality, separate):
    labels = label_sets(cardinality)
    query = labels[[0, 2]] if separate else labels
    reference = labels if separate else None
    expected_positive = []
    expected_negative = []
    for i, first in enumerate(query.tolist()):
        first_set = {k for k, present in enumerate(first) if present}
        for j, second in enumerate(labels.tolist()):
            if not separate and i == j:
                continue
            second_set = {k for k, present in enumerate(second) if present}
            target = expected_positive if first_set == second_set else expected_negative
            target.append((i, j))
    with torch.autocast("cpu", dtype=dtype):
        a, p, b, n = MultiSupConLoss(threshold=1).pair_indices(query, reference)
    assert list(zip(a.tolist(), p.tolist(), strict=True)) == expected_positive
    assert list(zip(b.tolist(), n.tolist(), strict=True)) == expected_negative


@pytest.mark.parametrize(
    "dtype,cardinality", [(torch.bfloat16, 257), (torch.float16, 2049)]
)
@pytest.mark.parametrize("supplied_pairs", [False, True])
def test_autocast_label_weights_preserve_loss_and_gradients(
    dtype, cardinality, supplied_pairs
):
    labels = label_sets(cardinality)
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]], requires_grad=True
    )
    loss = MultiSupConLoss(threshold=1)
    pairs = loss.pair_indices(labels) if supplied_pairs else None
    matrix = loss.distance(embeddings)
    reference = embeddings.detach().double().requires_grad_()
    expected = reference_loss(reference, labels.tolist(), threshold=1)
    expected_gradient = torch.autograd.grad(expected, reference)[0]
    with torch.autocast("cpu", dtype=dtype):
        actual = loss(embeddings, labels, pairs, mat=matrix)
    gradient = torch.autograd.grad(actual, embeddings)[0]
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(
        gradient.double(), expected_gradient, rtol=2e-5, atol=1e-7
    )


def test_autocast_label_sets_keep_memory_training_active():
    labels = label_sets(257)
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
    )
    loss = MultiSupConLoss(temperature=0.5, threshold=1)
    memory = CrossBatchMemory(
        loss,
        2,
        memory_size=8,
        label_dtype=torch.bool,
        label_shape=(514,),
        pair_function=loss.pair_indices,
    )
    expected = loss(embeddings, labels)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = memory(embeddings, labels)
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=1e-5)
    actual.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all() and embeddings.grad.abs().sum() > 0
    assert memory.queue_idx == 4
    assert torch.equal(memory.label_memory[:4], labels)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_keeps_empty_and_disjoint_label_set_contract(dtype):
    labels = torch.tensor(
        [[False, False], [False, False], [True, False], [False, True]]
    )
    with torch.autocast("cpu", dtype=dtype):
        a, p, _, _ = MultiSupConLoss(threshold=1).pair_indices(labels)
        empty = MultiSupConLoss().pair_indices(labels[:0])
    assert list(zip(a.tolist(), p.tolist(), strict=True)) == [(0, 1), (1, 0)]
    assert all(part.numel() == 0 for part in empty)


def test_label_counting_restores_the_callers_autocast_scope():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        MultiSupConLoss().pair_indices(label_sets(257))
        assert torch.is_autocast_enabled("cpu")
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.bfloat16
