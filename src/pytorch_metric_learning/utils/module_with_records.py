import torch

from . import common_functions as c_f


class ModuleWithRecords(torch.nn.Module):
    def __init__(self, collect_stats=None):
        super().__init__()
        self.collect_stats = (
            c_f.COLLECT_STATS if collect_stats is None else collect_stats
        )

    def register_stats(self, *names):
        if self.collect_stats:
            c_f.register_stats(self, names)

    def reset_stats(self):
        c_f.reset_stats(self)
