import logging

import torch

LOGGER_NAME = "PML"
LOGGER = logging.getLogger(LOGGER_NAME)
COLLECT_STATS = False


def set_logger_name(name):
    global LOGGER_NAME
    global LOGGER
    LOGGER_NAME = name
    LOGGER = logging.getLogger(LOGGER_NAME)


def pos_inf(dtype):
    return torch.finfo(dtype).max


def neg_inf(dtype):
    return torch.finfo(dtype).min


def small_val(dtype):
    return torch.finfo(dtype).tiny


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def numpy_to_torch(v):
    try:
        return torch.from_numpy(v)
    except TypeError:
        return v


def to_numpy(v):
    try:
        return v.cpu().numpy()
    except AttributeError:
        return v


def register_stats(input_obj, names):
    registered = input_obj.__dict__.setdefault("_record_these_stats", [])
    for name in names:
        if name not in registered:
            registered.append(name)
        if not hasattr(input_obj, name):
            setattr(input_obj, name, 0)


def reset_stats(input_obj):
    for name in getattr(input_obj, "_record_these_stats", []):
        setattr(input_obj, name, 0)


def check_shapes(embeddings, labels):
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def assert_distance_type(obj, distance_type=None, **kwargs):
    obj_name = type(obj).__name__
    if distance_type is not None:
        if is_list_or_tuple(distance_type):
            distance_type_str = ", ".join(x.__name__ for x in distance_type)
            distance_type_str = "one of " + distance_type_str
        else:
            distance_type_str = distance_type.__name__
        assert isinstance(obj.distance, distance_type), (
            f"{obj_name} requires the distance metric to be {distance_type_str}"
        )
    for k, v in kwargs.items():
        assert getattr(obj.distance, k) == v, (
            f"{obj_name} requires distance.{k} to be {v}"
        )


def check_parameter_placement(module, parameter, embeddings):
    same_device = parameter.device == embeddings.device
    autocast = torch.is_autocast_enabled(embeddings.device.type)
    if same_device and (autocast or parameter.dtype == embeddings.dtype):
        return
    raise ValueError(
        f"{type(module).__name__} parameters are {parameter.dtype} on "
        f"{parameter.device}, but embeddings are {embeddings.dtype} on "
        f"{embeddings.device}. Move the loss with .to(device, dtype) before "
        "calling it, and include its parameters in the optimizer."
    )


def torch_arange_from_size(input, size_dim=0):
    return torch.arange(input.size(size_dim), device=input.device)


class TorchInitWrapper:
    def __init__(self, init_func, **kwargs):
        self.init_func = init_func
        self.kwargs = kwargs

    def __call__(self, tensor):
        self.init_func(tensor, **self.kwargs)


def torch_all_from_dim_to_end(x, dim):
    return torch.all(x.view(*x.shape[:dim], -1), dim=-1)


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        if dtype is not None:
            dt = dtype
        elif tensor is not None:
            dt = tensor.dtype
        else:
            raise ValueError("to_dtype requires tensor or dtype")
        if x.dtype != dt:
            x = x.type(dt)
    return x


def to_device(x, tensor=None, device=None, dtype=None):
    if device is not None:
        dv = device
    elif tensor is not None:
        dv = tensor.device
    else:
        raise ValueError("to_device requires tensor or device")
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if ref_labels is not None:
            ref_labels = to_device(ref_labels, ref_emb)
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
        torch.cat([x, to_device(y, x)], dim=0) for x, y in zip(it1, it2, strict=True)
    )
