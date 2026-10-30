from collections.abc import Sequence

import torch
from torch import Tensor

from ..types import IndicesTuple, PairIndices, TripletIndices

LOW_PRECISION = (torch.float16, torch.bfloat16)


def pos_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).max


def neg_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def small_val(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).tiny


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in LOW_PRECISION else dtype


def normalization_eps(dtype: torch.dtype) -> float:
    return max(1e-12, torch.finfo(dtype).tiny)


def normalize(embeddings: Tensor, p: float = 2, dim: int = 1) -> Tensor:
    work = embeddings.to(accumulation_dtype(embeddings.dtype))
    eps = normalization_eps(embeddings.dtype)
    normalized = torch.nn.functional.normalize(work, p=p, dim=dim, eps=eps)
    return normalized.to(embeddings.dtype)


INDEX_DTYPES = (torch.int64, torch.int32)


def check_index_dtypes(tensors: Sequence[Tensor], name: str) -> None:
    if any(x.dtype not in INDEX_DTYPES for x in tensors):
        raise TypeError(f"{name} must be int64 or int32 tensors")


def empty_pairs(device: torch.device) -> PairIndices:
    empty = torch.empty(0, dtype=torch.long, device=device)
    return PairIndices(empty, empty, empty, empty)


def empty_triplets(device: torch.device) -> TripletIndices:
    empty = torch.empty(0, dtype=torch.long, device=device)
    return TripletIndices(empty, empty, empty)


def zero_loss(embeddings: Tensor) -> Tensor:
    return embeddings.reshape(-1)[:0].sum()


def check_shapes(embeddings: Tensor, labels: Tensor | None) -> None:
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch,)")
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")


def check_device(tensor: Tensor | None, embeddings: Tensor, name: str) -> None:
    if tensor is not None and tensor.device != embeddings.device:
        raise ValueError(
            f"{name} are on {tensor.device}, but embeddings are on "
            f"{embeddings.device}. Move them to the embeddings' device."
        )


def check_distance_type(
    obj: torch.nn.Module, distance_type: type | tuple[type, ...], **attributes
) -> None:
    if not isinstance(obj.distance, distance_type):
        types = distance_type if isinstance(distance_type, tuple) else (distance_type,)
        names = " or ".join(t.__name__ for t in types)
        raise TypeError(f"{type(obj).__name__} requires the distance to be {names}")
    for name, value in attributes.items():
        if getattr(obj.distance, name) != value:
            raise ValueError(
                f"{type(obj).__name__} requires distance.{name} to be {value}"
            )


def check_parameter_placement(
    module: torch.nn.Module, parameter: Tensor, embeddings: Tensor
) -> None:
    same_device = parameter.device == embeddings.device
    autocast = torch.is_autocast_enabled(embeddings.device.type)
    if same_device and (autocast or parameter.dtype == embeddings.dtype):
        return
    raise ValueError(
        f"{type(module).__name__} tensors are {parameter.dtype} on "
        f"{parameter.device}, but embeddings are {embeddings.dtype} on "
        f"{embeddings.device}. Move the loss with .to(device, dtype) before "
        "calling it."
    )


def arange_like(x: Tensor, dim: int = 0) -> Tensor:
    return torch.arange(x.size(dim), device=x.device)


def prepare_inputs(
    embeddings: Tensor,
    labels: Tensor | None,
    ref_emb: Tensor | None,
    ref_labels: Tensor | None,
) -> None:
    check_shapes(embeddings, labels)
    check_device(labels, embeddings, "labels")
    if ref_emb is None:
        if ref_labels is not None:
            raise ValueError("ref_labels require ref_emb")
        return
    check_shapes(ref_emb, ref_labels)
    check_device(ref_emb, embeddings, "ref_emb")
    check_device(ref_labels, embeddings, "ref_labels")
    if (labels is None) != (ref_labels is None):
        raise ValueError(
            "labels and ref_labels must both be supplied or both omitted with ref_emb"
        )


def resolve_reference(
    embeddings: Tensor,
    labels: Tensor | None,
    ref_emb: Tensor | None,
    ref_labels: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    if ref_emb is None:
        return embeddings, labels
    return ref_emb, ref_labels


def as_indices_tuple(
    indices_tuple: Sequence[Tensor] | None, device: torch.device | None = None
) -> IndicesTuple | None:
    if indices_tuple is None:
        return None
    if not isinstance(indices_tuple, (tuple, list)) or not all(
        isinstance(x, Tensor) for x in indices_tuple
    ):
        raise TypeError("indices_tuple must be a tuple of index tensors")
    check_index_dtypes(indices_tuple, "indices_tuple")
    if any(x.ndim != 1 for x in indices_tuple):
        raise ValueError("indices_tuple must contain one-dimensional tensors")
    if device is not None and any(x.device != device for x in indices_tuple):
        raise ValueError("indices_tuple tensors must be on the embeddings' device")
    if len(indices_tuple) == 3:
        if not len(indices_tuple[0]) == len(indices_tuple[1]) == len(indices_tuple[2]):
            raise ValueError("Triplet indices must contain three equal-length tensors")
        return TripletIndices(*indices_tuple)
    if len(indices_tuple) == 4:
        a1, p, a2, n = indices_tuple
        if len(a1) != len(p) or len(a2) != len(n):
            raise ValueError(
                "Pair indices must have equal-length anchor and partner tensors"
            )
        return PairIndices(*indices_tuple)
    raise ValueError("indices_tuple must contain three triplet or four pair tensors")


def ref_not_supported(ref_emb: Tensor | None) -> None:
    if ref_emb is not None:
        raise ValueError("ref_emb is not supported for this loss function")


def indices_tuple_not_supported(indices_tuple: object) -> None:
    if indices_tuple is not None:
        raise ValueError("indices_tuple is not supported for this loss function")


def labels_required(labels: Tensor | None) -> Tensor:
    if labels is None:
        raise ValueError("labels are required for this loss function")
    return labels


def labels_or_indices_tuple_required(labels: object, indices_tuple: object) -> None:
    if labels is None and indices_tuple is None:
        raise ValueError("labels and indices_tuple cannot both be None")


def logsumexp(
    x: Tensor, keep_mask: Tensor | None = None, add_one: bool = True, dim: int = 1
) -> Tensor:
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, neg_inf(x.dtype))
    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if add_one:
        output = torch.logaddexp(output, torch.zeros_like(output))
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output


def segment_logsumexp(values: Tensor, segments: Tensor, num_segments: int) -> Tensor:
    work = values.to(accumulation_dtype(values.dtype))
    maxima = work.new_full((num_segments,), -torch.inf)
    maxima = maxima.scatter_reduce(0, segments, work.detach(), reduce="amax")
    shift = torch.where(torch.isfinite(maxima), maxima, 0)
    sums = work.new_zeros(num_segments).index_add(
        0, segments, torch.exp(work - shift[segments])
    )
    present = sums > 0
    output = shift + torch.log(torch.where(present, sums, 1))
    return torch.where(present, output, -torch.inf).to(values.dtype)
