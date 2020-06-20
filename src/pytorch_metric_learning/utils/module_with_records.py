import torch
from . import common_functions as c_f


class ModuleWithRecords(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)

    def reset_stats(self):
        c_f.reset_stats(self)
