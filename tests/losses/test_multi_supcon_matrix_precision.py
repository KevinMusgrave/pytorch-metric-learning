import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses, reducers
from pytorch_metric_learning.types import PairIndices

from .. import TEST_DEVICE, TEST_DTYPES


def matrix_reference(mat, labels, references, pairs, temperature, inverted):
    pos = set(zip(pairs.pos_anchors.tolist(), pairs.positives.tolist(), strict=True))
    neg = set(zip(pairs.neg_anchors.tolist(), pairs.negatives.tolist(), strict=True))
    query_sets = [
        {i for i, value in enumerate(row) if value} for row in labels.tolist()
    ]
    ref_sets = [
        {i for i, value in enumerate(row) if value} for row in references.tolist()
    ]
    rows = []
    for anchor in range(len(labels)):
        positive = sorted(j for i, j in pos if i == anchor)
        if not positive:
            continue
        compared = sorted({j for i, j in pos | neg if i == anchor})
        logits = mat[anchor, compared] * (1 if inverted else -1) / temperature
        log_probability = F.log_softmax(logits, dim=0)
        terms = []
        for index in positive:
            first, second = query_sets[anchor], ref_sets[index]
            overlap = len(first & second) / len(first | second) if first | second else 1
            terms.append(-overlap * log_probability[compared.index(index)])
        rows.append(torch.stack(terms).mean())
    return torch.stack(rows).mean()


@pytest.mark.parametrize("embedding_dtype", [*TEST_DTYPES, torch.bfloat16])
@pytest.mark.parametrize("offset", [0.0, 1e8])
@pytest.mark.parametrize("separate_reference", [False, True])
@pytest.mark.parametrize("supplied_pairs", [False, True])
@pytest.mark.parametrize("inverted", [False, True])
def test_multisupcon_preserves_double_matrix_values_and_gradients(
    embedding_dtype, offset, separate_reference, supplied_pairs, inverted
):
    labels = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
        dtype=torch.bool,
        device=TEST_DEVICE,
    )
    references = (
        torch.tensor(
            [[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0]],
            dtype=torch.bool,
            device=TEST_DEVICE,
        )
        if separate_reference
        else labels
    )
    embeddings = torch.ones(4, 2, dtype=embedding_dtype, device=TEST_DEVICE)
    reference_embeddings = (
        torch.ones(5, 2, dtype=embedding_dtype, device=TEST_DEVICE)
        if separate_reference
        else None
    )
    distance = (
        distances.DotProductSimilarity(normalize_embeddings=False)
        if inverted
        else distances.LpDistance(normalize_embeddings=False)
    )
    loss = losses.MultiSupConLoss(
        temperature=0.7,
        threshold=0.5,
        distance=distance,
        reducer=reducers.MeanReducer(),
    )
    pairs = loss.pair_indices(labels, references if separate_reference else None)
    if supplied_pairs:
        pairs = PairIndices(
            *(
                torch.tensor(x, device=TEST_DEVICE)
                for x in ([0, 2], [1, 3], [0, 2], [2, 0])
            )
        )
    matrix = (
        torch.arange(
            4 * len(references), dtype=torch.float64, device=TEST_DEVICE
        ).reshape(4, -1)
        % 7
    ) / 10 + offset
    actual_matrix = matrix.clone().requires_grad_()
    reference_matrix = matrix.clone().requires_grad_()
    actual = loss(
        embeddings,
        labels,
        pairs if supplied_pairs else None,
        ref_emb=reference_embeddings,
        ref_labels=references if separate_reference else None,
        mat=actual_matrix,
    )
    expected = matrix_reference(
        reference_matrix, labels, references, pairs, 0.7, inverted
    )
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=3e-7, atol=1e-9)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(
        actual_matrix.grad, reference_matrix.grad, rtol=3e-7, atol=1e-9
    )
    torch.testing.assert_close(actual_matrix.detach(), matrix)
