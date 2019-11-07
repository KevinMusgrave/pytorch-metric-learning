import collections
import torch
from torch.autograd import Variable
import numpy as np



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


def apply_func_to_dict(input, f):
    if isinstance(input, collections.Mapping):
        for k, v in input.items():
            input[k] = f(v)
        return input
    else:
        return f(input)


def wrap_variable(batch_data, device):
    def f(x):
        return Variable(x).to(device)

    return apply_func_to_dict(batch_data, f)


def get_hierarchy_label(batch_labels, hierarchy_level):
    def f(v):
        try:
            if v.ndim == 2:
                v = v[:, hierarchy_level]
            return v
        except BaseException:
            return v

    return apply_func_to_dict(batch_labels, f)


def numpy_to_torch(input):
    def f(v):
        try:
            return torch.from_numpy(v)
        except BaseException:
            return v

    return apply_func_to_dict(input, f)


def torch_to_numpy(input):
    def f(v):
        try:
            return v.cpu().numpy()
        except BaseException:
            return v

    return apply_func_to_dict(input, f)


def process_label(labels, hierarchy_level, label_map):
    labels = get_hierarchy_label(labels, hierarchy_level)
    labels = torch_to_numpy(labels)
    labels = label_map(labels, hierarchy_level)
    labels = numpy_to_torch(labels)
    return labels


def pass_data_to_model(model, data, device, **kwargs):
    if isinstance(data, collections.Mapping):
        base_output = {}
        for k, v in data.items():
            base_output[k] = model(wrap_variable(v, device), k=k, **kwargs)
        return base_output
    else:
        return model(wrap_variable(data, device), **kwargs)

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad



def copy_params_to_another_model(from_model, to_model):
    params1 = from_model.named_parameters()
    params2 = to_model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data)


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
    return np.random.choice(input_data, size=size, replace=replace)


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


def get_dataloader(dataset, batch_size, sampler, num_workers, collate_fn):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=sampler is None
    )


def try_torch_operation(torch_op, input_val):
    return torch_op(input_val) if torch.is_tensor(input_val) else input_val 