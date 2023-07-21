from .base_reducer import BaseReducer


class DoNothingReducer(BaseReducer):
    def forward(self, loss_dict, *_):
        return loss_dict
