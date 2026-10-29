import torch
from torch import Tensor

LOW_PRECISION = (torch.float16, torch.bfloat16)


def pos_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).max


def neg_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def small_val(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).tiny


def zero_loss(embeddings: Tensor) -> Tensor:
    return embeddings.reshape(-1)[:0].sum()


def check_shapes(embeddings: Tensor, labels: Tensor | None) -> None:
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch,)")
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")


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


def all_from_dim(x: Tensor, dim: int) -> Tensor:
    return torch.all(x.reshape(*x.shape[:dim], -1), dim=-1)


def prepare_inputs(
    embeddings: Tensor,
    labels: Tensor | None,
    ref_emb: Tensor | None,
    ref_labels: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
    check_shapes(embeddings, labels)
    if labels is not None:
        labels = labels.to(device=embeddings.device)
    if ref_emb is None:
        ref_emb, ref_labels = embeddings, labels
    elif ref_labels is not None:
        ref_labels = ref_labels.to(device=ref_emb.device)
    check_shapes(ref_emb, ref_labels)
    return embeddings, labels, ref_emb, ref_labels


def labels_not_supported(labels: Tensor | None, ref_labels: Tensor | None) -> None:
    if labels is not None or ref_labels is not None:
        raise ValueError("labels and ref_labels are not supported for this loss")


def ref_not_supported(
    embeddings: Tensor,
    labels: Tensor | None,
    ref_emb: Tensor,
    ref_labels: Tensor | None,
) -> None:
    if ref_emb is not embeddings or ref_labels is not labels:
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
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)
    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output
