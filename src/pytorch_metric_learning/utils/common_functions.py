import collections
import logging
import urllib

import numpy as np
import scipy.stats
import torch
from tqdm import tqdm

LOGGER_NAME = "PML"
LOGGER = logging.getLogger(LOGGER_NAME)
NUMPY_RANDOM = np.random
COLLECT_STATS = False


# taken from:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py#L27
def _urlretrieve(url, filename, chunk_size=1024 * 32):
    with urllib.request.urlopen(urllib.request.Request(url)) as response:
        with (
            open(filename, "wb") as fh,
            tqdm(total=response.length, unit="B", unit_scale=True) as pbar,
        ):
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))


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


def try_next_on_generator(gen, iterable):
    try:
        return gen, next(gen)
    except StopIteration:
        gen = iter(iterable)
        return gen, next(gen)


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


def get_hierarchy_label(batch_labels, hierarchy_level):
    if hierarchy_level == "all":
        return batch_labels
    if is_list_or_tuple(hierarchy_level):
        max_hierarchy_level = max(hierarchy_level)
    else:
        max_hierarchy_level = hierarchy_level
    if max_hierarchy_level > 0:
        assert (batch_labels.ndim == 2) and batch_labels.shape[1] > max_hierarchy_level
    if batch_labels.ndim == 2:
        batch_labels = batch_labels[:, hierarchy_level]
    return batch_labels


def map_labels(label_map, labels):
    if is_list_or_tuple(labels):
        labels = np.stack([to_numpy(sub) for sub in labels], axis=1)
    else:
        labels = to_numpy(labels)
    if labels.ndim == 2:
        new_labels = np.zeros(labels.shape, dtype=int)
        for h in range(labels.shape[1]):
            new_labels[:, h] = label_map(labels[:, h], h)
    else:
        new_labels = label_map(labels, 0)
    return new_labels


def process_label(labels, hierarchy_level, label_map):
    labels = map_labels(label_map, labels)
    labels = get_hierarchy_label(labels, hierarchy_level)
    labels = numpy_to_torch(labels)
    return labels


def safe_random_choice(input_data, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return NUMPY_RANDOM.choice(input_data, size=size, replace=replace)


def get_eval_dataloader(dataset, batch_size, num_workers, collate_fn):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False,
    )


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices


def make_label_to_rank_dict(label_set):
    """
    Args:
        label_set: type sequence, a set of integer labels
                    (no duplicates in the sequence)
    Returns:
        A dictionary mapping each label to its numeric rank in the original set
    """
    if len(set(label_set)) != len(label_set):
        raise ValueError("label set must not have duplicates")
    ranked = scipy.stats.rankdata(label_set).astype(int) - 1
    return {k: v for k, v in zip(label_set, ranked)}


def get_label_map(labels):
    # Returns a nested dictionary.
    # First level of dictionary represents label hierarchy level.
    # Second level is the label map for that hierarchy level
    labels = np.array(labels)
    if labels.ndim == 2:
        label_map = {}
        for hierarchy_level in range(labels.shape[1]):
            label_map[hierarchy_level] = make_label_to_rank_dict(
                list(set(labels[:, hierarchy_level]))
            )
        return label_map
    return {0: make_label_to_rank_dict(list(set(labels)))}


class LabelMapper:
    def __init__(self, set_min_label_to_zero=False, dataset_labels=None):
        self.set_min_label_to_zero = set_min_label_to_zero
        if dataset_labels is not None:
            self.label_map = get_label_map(dataset_labels)

    def map(self, labels, hierarchy_level):
        if not self.set_min_label_to_zero:
            return labels
        else:
            return np.array(
                [self.label_map[hierarchy_level][x] for x in labels], dtype=int
            )


def add_to_recordable_attributes(
    input_obj, name=None, list_of_names=None, is_stat=False
):
    if is_stat:
        attr_name_list_name = "_record_these_stats"
    else:
        attr_name_list_name = "_record_these"
    if not hasattr(input_obj, attr_name_list_name):
        setattr(input_obj, attr_name_list_name, [])
    attr_name_list = getattr(input_obj, attr_name_list_name)
    if name is not None:
        if name not in attr_name_list:
            attr_name_list.append(name)
        if not hasattr(input_obj, name):
            setattr(input_obj, name, 0)
    if list_of_names is not None and is_list_or_tuple(list_of_names):
        for n in list_of_names:
            add_to_recordable_attributes(input_obj, name=n, is_stat=is_stat)


def reset_stats(input_obj):
    for attr_list in ["_record_these_stats"]:
        for r in getattr(input_obj, attr_list, []):
            setattr(input_obj, r, 0)


def return_input(x):
    return x


def check_shapes(embeddings, labels):
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def assert_distance_type(obj, distance_type=None, **kwargs):
    if distance_type is not None:
        if is_list_or_tuple(distance_type):
            distance_type_str = ", ".join(x.__name__ for x in distance_type)
            distance_type_str = "one of " + distance_type_str
        else:
            distance_type_str = distance_type.__name__
        obj_name = obj.__class__.__name__
        assert isinstance(obj.distance, distance_type), (
            "{} requires the distance metric to be {}".format(
                obj_name, distance_type_str
            )
        )
    for k, v in kwargs.items():
        assert getattr(obj.distance, k) == v, "{} requires distance.{} to be {}".format(
            obj_name, k, v
        )


def torch_arange_from_size(input, size_dim=0):
    return torch.arange(input.size(size_dim), device=input.device)


class TorchInitWrapper:
    def __init__(self, init_func, **kwargs):
        self.init_func = init_func
        self.kwargs = kwargs

    def __call__(self, tensor):
        self.init_func(tensor, **self.kwargs)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def torch_all_from_dim_to_end(x, dim):
    return torch.all(x.view(*x.shape[:dim], -1), dim=-1)


def torch_standard_scaler(x):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    return (x - mean) / std


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x


def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
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
    return tuple([torch.cat([x, to_device(y, x)], dim=0) for x, y in zip(it1, it2)])


def use_cuda_if_available():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
