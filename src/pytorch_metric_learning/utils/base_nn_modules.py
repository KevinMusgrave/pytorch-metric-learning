import torch
from . import common_functions as c_f
from ..reducers import MeanReducer, MultipleReducers


class ModuleWithStats(torch.nn.Module):
    def __init__(self, collect_stats=True):
        super().__init__()
        self.collect_stats = collect_stats

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False, optional=False):
        if not optional or self.collect_stats: 
            c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)


class ModuleWithStatsAndReducer(ModuleWithStats):
    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducer = self.get_reducer() if reducer is None else reducer

    def get_default_reducer(self):
        return MeanReducer()

    def get_reducer(self):
        reducer = self.get_default_reducer()
        if isinstance(reducer, MultipleReducers) or len(self.sub_loss_names()) == 1:
            return reducer
        return MultipleReducers({k:self.get_default_reducer() for k in self.sub_loss_names()})

    def sub_loss_names(self):
        return ["loss"]