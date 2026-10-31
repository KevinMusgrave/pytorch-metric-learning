from dataclasses import dataclass

import torch
from torch import Tensor

from ..reducers import AvgNonZeroReducer, BaseReducer, MeanReducer
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


@dataclass
class _Term:
    total: Tensor
    count: int
    gradients: list[Tensor | None]


def anchor_batched_backward(
    loss_func: BaseMetricLossFunction,
    embeddings: Tensor,
    labels: Tensor,
    batch_size: int,
    *,
    ref_emb: Tensor | None = None,
    ref_labels: Tensor | None = None,
) -> Tensor:
    _check_supported(loss_func, batch_size)
    _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels)
    outputs = [x for x in (embeddings, ref_emb) if x is not None and x.requires_grad]
    if not outputs or not torch.is_grad_enabled():
        raise ValueError("anchor_batched_backward needs inputs tracking gradients")
    dtype = _common.accumulation_dtype(embeddings.dtype)
    query = embeddings.detach().to(dtype).requires_grad_()
    reference = (
        query if ref_emb is None else ref_emb.detach().to(dtype).requires_grad_()
    )
    reference_labels = labels if ref_labels is None else ref_labels
    parameters = [p for p in loss_func.parameters() if p.requires_grad]
    inputs = (embeddings,) if ref_emb is None else (embeddings, ref_emb)
    copies = (query,) if ref_emb is None else (query, reference)
    leaves = (*copies, *parameters)
    terms: dict[str, _Term] = {}
    for rows in torch.arange(len(query), device=query.device).split(batch_size):
        pairs = indexing.get_all_pairs_indices(labels[rows], reference_labels)
        if ref_emb is None:
            pairs = indexing._remove_self_comparisons(pairs, rows)
        chunk = loss_func.loss_terms(
            query[rows], labels[rows], pairs, reference, reference_labels
        )
        _accumulate(terms, chunk, loss_func.reducer, leaves)
        del chunk
    gradients = [_reduced_gradient(terms, i) for i in range(len(leaves))]
    if gradients[0] is None:
        gradients[0] = torch.zeros_like(query)
    tracked = [
        (x, g.to(x.dtype))
        for x, g in zip((*inputs, *parameters), gradients, strict=True)
        if x.requires_grad and g is not None
    ]
    torch.autograd.backward([x for x, _ in tracked], [g for _, g in tracked])
    total = sum(
        (t.total / max(t.count, 1) for t in terms.values()), start=query.new_zeros(())
    )
    return total.to(embeddings.dtype)


def _reduced_gradient(terms: dict[str, _Term], index: int) -> Tensor | None:
    reduced = [
        gradient / max(term.count, 1)
        for term in terms.values()
        if (gradient := term.gradients[index]) is not None
    ]
    return sum(reduced[1:], start=reduced[0]) if reduced else None


def _check_supported(loss_func: BaseMetricLossFunction, batch_size: int) -> None:
    name = type(loss_func).__name__
    if not getattr(loss_func, "decomposes_by_anchor", False):
        raise ValueError(
            f"{name} cannot be batched by anchor: its objective is not a sum of "
            "per-anchor terms"
        )
    if type(loss_func.reducer) not in (MeanReducer, AvgNonZeroReducer):
        raise ValueError(
            "anchor_batched_backward supports only MeanReducer and AvgNonZeroReducer"
        )
    if loss_func.embedding_regularizer is not None:
        raise ValueError("anchor_batched_backward does not support regularizers")
    if getattr(loss_func, "triplets_per_anchor", "all") != "all":
        raise ValueError('anchor_batched_backward requires triplets_per_anchor="all"')
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _accumulate(
    terms: dict[str, _Term],
    chunk: dict,
    reducer: BaseReducer,
    leaves: tuple[Tensor, ...],
) -> None:
    for name, term in chunk.items():
        losses = term["losses"]
        if losses.numel() == 0:
            continue
        kept = losses if isinstance(reducer, MeanReducer) else losses[losses > 0]
        total = kept.sum()
        gradients = torch.autograd.grad(
            total, leaves, retain_graph=True, allow_unused=True
        )
        count = kept.numel()
        if name not in terms:
            terms[name] = _Term(total.detach(), count, list(gradients))
            continue
        stored = terms[name]
        stored.total = stored.total + total.detach()
        stored.count += count
        stored.gradients = [
            x if y is None else y if x is None else x + y
            for x, y in zip(stored.gradients, gradients, strict=True)
        ]
