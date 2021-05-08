import collections
import glob
import logging
import os
import re

import numpy as np
import scipy.stats
import torch

LOGGER_NAME = "PML"
LOGGER = logging.getLogger(LOGGER_NAME)
NUMPY_RANDOM = np.random
COLLECT_STATS = True


def set_logger_name(name):
    global LOGGER_NAME
    global LOGGER
    LOGGER_NAME = name
    LOGGER = logging.getLogger(LOGGER_NAME)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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
    if is_list_or_tuple(v):
        return np.stack([to_numpy(sub_v) for sub_v in v], axis=1)
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
    labels = to_numpy(labels)
    if labels.ndim == 2:
        for h in range(labels.shape[1]):
            labels[:, h] = label_map(labels[:, h], h)
    else:
        labels = label_map(labels, 0)
    return labels


def process_label(labels, hierarchy_level, label_map):
    labels = map_labels(label_map, labels)
    labels = get_hierarchy_label(labels, hierarchy_level)
    labels = numpy_to_torch(labels)
    return labels


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def shift_indices_tuple(indices_tuple, batch_size):
    """
    Shifts indices of positives and negatives of pairs or triplets by batch_size

    if len(indices_tuple) != 3 or len(indices_tuple) != 4, it will return indices_tuple
    Args:
        indices_tuple is a tuple with torch.Tensor
        batch_size is an int
    Returns:
        A tuple with shifted indices
    """

    if len(indices_tuple) == 3:
        indices_tuple = (indices_tuple[0],) + tuple(
            [x + batch_size if len(x) > 0 else x for x in indices_tuple[1:]]
        )
    elif len(indices_tuple) == 4:
        indices_tuple = tuple(
            [
                x + batch_size if len(x) > 0 and i % 2 == 1 else x
                for i, x in enumerate(indices_tuple)
            ]
        )
    return indices_tuple


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


def longest_list(list_of_lists):
    return max(list_of_lists, key=len)


def slice_by_n(input_array, n):
    output = []
    for i in range(n):
        output.append(input_array[i::n])
    return output


def unslice_by_n(input_tensors):
    n = len(input_tensors)
    rows, cols = input_tensors[0].size()
    output = torch.zeros((rows * n, cols), device=input_tensors[0].device)
    for i in range(n):
        output[i::n] = input_tensors[i]
    return output


def set_layers_to_eval(layer_name):
    def set_to_eval(m):
        classname = m.__class__.__name__
        if classname.find(layer_name) != -1:
            m.eval()

    return set_to_eval


def get_train_dataloader(dataset, batch_size, sampler, num_workers, collate_fn):
    if isinstance(sampler, torch.utils.data.BatchSampler):
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=sampler is None,
        pin_memory=False,
    )


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


def try_torch_operation(torch_op, input_val):
    return torch_op(input_val) if torch.is_tensor(input_val) else input_val


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
        labels_to_indices[k] = np.array(v, dtype=np.int)
    return labels_to_indices


def make_label_to_rank_dict(label_set):
    """
    Args:
        label_set: type sequence, a set of integer labels
                    (no duplicates in the sequence)
    Returns:
        A dictionary mapping each label to its numeric rank in the original set
    """
    ranked = scipy.stats.rankdata(label_set) - 1
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
                [self.label_map[hierarchy_level][x] for x in labels], dtype=np.int
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


def list_of_recordable_attributes_list_names():
    return ["_record_these", "_record_these_stats"]


def modelpath_creator(folder, basename, identifier, extension=".pth"):
    if identifier is None:
        return os.path.join(folder, basename + extension)
    else:
        return os.path.join(folder, "%s_%s%s" % (basename, str(identifier), extension))


def save_model(model, filepath):
    if any(
        isinstance(model, x)
        for x in [torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel]
    ):
        torch.save(model.module.state_dict(), filepath)
    else:
        torch.save(model.state_dict(), filepath)


def load_model(model_def, model_filename, device):
    try:
        model_def.load_state_dict(torch.load(model_filename, map_location=device))
    except KeyError:
        # original saved file with DataParallel
        state_dict = torch.load(model_filename)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model_def.load_state_dict(new_state_dict)


def operate_on_dict_of_models(
    input_dict,
    suffix,
    folder,
    operation,
    logging_string="",
    log_if_successful=False,
    assert_success=False,
):
    for k, v in input_dict.items():
        model_path = modelpath_creator(folder, k, suffix)
        try:
            operation(v, model_path)
            if log_if_successful:
                LOGGER.info("%s %s" % (logging_string, model_path))
        except IOError:
            LOGGER.warning("Could not %s %s" % (logging_string, model_path))
            if assert_success:
                raise IOError


def save_dict_of_models(input_dict, suffix, folder, **kwargs):
    def operation(v, model_path):
        save_model(v, model_path)

    operate_on_dict_of_models(input_dict, suffix, folder, operation, "SAVE", **kwargs)


def load_dict_of_models(input_dict, suffix, folder, device, **kwargs):
    def operation(v, model_path):
        load_model(v, model_path, device)

    operate_on_dict_of_models(input_dict, suffix, folder, operation, "LOAD", **kwargs)


def delete_dict_of_models(input_dict, suffix, folder, **kwargs):
    def operation(v, model_path):
        if os.path.exists(model_path):
            os.remove(model_path)

    operate_on_dict_of_models(input_dict, suffix, folder, operation, "DELETE", **kwargs)


def regex_wrapper(x):
    if isinstance(x, list):
        return [re.compile(z) for z in x]
    return re.compile(x)


def regex_replace(search, replace, contents):
    return re.sub(search, replace, contents)


def latest_version(folder, string_to_glob="trunk_*.pth", best=False):
    items = glob.glob(os.path.join(folder, string_to_glob))
    if items == []:
        return (0, None)
    model_regex = (
        regex_wrapper("best[0-9]+\.pth$") if best else regex_wrapper("[0-9]+\.pth$")
    )
    epoch_regex = regex_wrapper("[0-9]+\.pth$")
    items = [x for x in items if model_regex.search(x)]
    version = [int(epoch_regex.findall(x)[-1].split(".")[0]) for x in items]
    resume_epoch = max(version)
    suffix = "best%d" % resume_epoch if best else resume_epoch
    return resume_epoch, suffix


def return_input(x):
    return x


def angle_to_coord(angle):
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    return x, y


def check_shapes(embeddings, labels):
    if embeddings.size(0) != labels.size(0):
        raise ValueError("Number of embeddings must equal number of labels")
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
        )
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def assert_distance_type(obj, distance_type=None, **kwargs):
    if distance_type is not None:
        if is_list_or_tuple(distance_type):
            distance_type_str = ", ".join(x.__name__ for x in distance_type)
            distance_type_str = "one of " + distance_type_str
        else:
            distance_type_str = distance_type.__name__
        obj_name = obj.__class__.__name__
        assert isinstance(
            obj.distance, distance_type
        ), "{} requires the distance metric to be {}".format(
            obj_name, distance_type_str
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


def sqlite_obj_to_dict(sqlite_obj):
    return {k: [row[k] for row in sqlite_obj] for k in sqlite_obj[0].keys()}


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
        ref_labels = to_device(ref_labels, ref_emb)
    else:
        ref_emb, ref_labels = embeddings, labels
    check_shapes(ref_emb, ref_labels)
    return ref_emb, ref_labels
