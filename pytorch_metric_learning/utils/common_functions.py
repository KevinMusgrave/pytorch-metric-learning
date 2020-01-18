import collections
import torch
from torch.autograd import Variable
import numpy as np

NUMPY_RANDOM_STATE = np.random.RandomState()


def try_keys(input_dict, keys):
    for k in keys:
        try:
            return input_dict[k]
        except BaseException:
            pass
    return None


def try_next_on_generator(gen, iterable):
    try:
        return gen, next(gen)
    except StopIteration:
        gen = iter(iterable)
        return gen, next(gen)


def numpy_to_torch(v):
    try:
        return torch.from_numpy(v)
    except BaseException:
        return v

def torch_to_numpy(v):
    try:
        return v.cpu().numpy()
    except BaseException:
        return v


def wrap_variable(batch_data, device):
    return Variable(batch_data).to(device)


def get_hierarchy_label(batch_labels, hierarchy_level):
    if hierarchy_level == "all":
        return batch_labels

    try:
        if batch_labels.ndim == 2:
            batch_labels = batch_labels[:, hierarchy_level]
        return batch_labels
    except BaseException:
        return batch_labels


def map_labels(label_map, labels):
    labels = torch_to_numpy(labels)
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

def pass_data_to_model(model, data, device, **kwargs):
    return model(wrap_variable(data, device), **kwargs)

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


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
    return NUMPY_RANDOM_STATE.choice(input_data, size=size, replace=replace)


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
    output = torch.zeros((rows * n, cols)).to(input_tensors[0].device)
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=sampler is None,
        pin_memory=False
    )

def get_eval_dataloader(dataset, batch_size, num_workers, collate_fn):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False
    )


def try_torch_operation(torch_op, input_val):
    return torch_op(input_val) if torch.is_tensor(input_val) else input_val 


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
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
    argsorted = list(np.argsort(label_set))
    return {k: v for k, v in zip(label_set, argsorted)}


def get_label_map(labels):
    # Returns a nested dictionary. 
    # First level of dictionary represents label hierarchy level.
    # Second level is the label map for that hierarchy level
    labels = np.array(labels)
    if labels.ndim == 2:
        label_map = {}
        for hierarchy_level in range(labels.shape[1]):
            label_map[hierarchy_level] = make_label_to_rank_dict(list(set(labels[:, hierarchy_level])))
        return label_map
    return {0: make_label_to_rank_dict(list(set(labels)))} 