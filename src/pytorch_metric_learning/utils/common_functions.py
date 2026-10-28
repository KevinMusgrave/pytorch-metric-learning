import torch


def pos_inf(dtype):
    return torch.finfo(dtype).max


def neg_inf(dtype):
    return torch.finfo(dtype).min


def small_val(dtype):
    return torch.finfo(dtype).tiny


def check_shapes(embeddings, labels):
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def check_distance_type(obj, distance_type=None, **kwargs):
    obj_name = type(obj).__name__
    if distance_type is not None:
        if isinstance(distance_type, (list, tuple)):
            distance_type = tuple(distance_type)
            distance_type_str = ", ".join(x.__name__ for x in distance_type)
            distance_type_str = "one of " + distance_type_str
        else:
            distance_type_str = distance_type.__name__
        if not isinstance(obj.distance, distance_type):
            raise TypeError(
                f"{obj_name} requires the distance metric to be {distance_type_str}"
            )
    for k, v in kwargs.items():
        if getattr(obj.distance, k) != v:
            raise ValueError(f"{obj_name} requires distance.{k} to be {v}")


def check_parameter_placement(module, parameter, embeddings):
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


def torch_arange_from_size(input, size_dim=0):
    return torch.arange(input.size(size_dim), device=input.device)


def torch_all_from_dim_to_end(x, dim):
    return torch.all(x.view(*x.shape[:dim], -1), dim=-1)


def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if ref_labels is not None:
            ref_labels = ref_labels.to(device=ref_emb.device)
    else:
        ref_emb, ref_labels = embeddings, labels
    check_shapes(ref_emb, ref_labels)
    return ref_emb, ref_labels


def labels_not_supported(labels, ref_labels):
    if labels is not None or ref_labels is not None:
        raise ValueError(
            "labels are ref_labels are not supported for this loss function"
        )


def ref_not_supported(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not embeddings or ref_labels is not labels:
        raise ValueError("ref_emb is not supported for this loss function")


def indices_tuple_not_supported(indices_tuple):
    if indices_tuple is not None:
        raise ValueError("indices_tuple is not supported for this loss function")


def labels_required(labels):
    if labels is None:
        raise ValueError("labels are required for this loss function")


def labels_or_indices_tuple_required(labels, indices_tuple):
    if labels is None and indices_tuple is None:
        raise ValueError("labels and indices_tuple cannot both be None")


def concatenate_indices_tuples(it1, it2):
    return tuple(
        torch.cat([x, y.to(device=x.device)], dim=0)
        for x, y in zip(it1, it2, strict=True)
    )
