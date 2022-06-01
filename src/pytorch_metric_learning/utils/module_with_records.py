import torch

from . import common_functions as c_f


class ModuleWithRecords(torch.nn.Module):
    def __init__(self, collect_stats=None):
        super().__init__()
        self.collect_stats = (
            c_f.COLLECT_STATS if collect_stats is None else collect_stats
        )

    def add_to_recordable_attributes(
        self, name=None, list_of_names=None, is_stat=False
    ):
        if is_stat and not self.collect_stats:
            pass
        else:
            c_f.add_to_recordable_attributes(
                self, name=name, list_of_names=list_of_names, is_stat=is_stat
            )

    def reset_stats(self):
        c_f.reset_stats(self)
