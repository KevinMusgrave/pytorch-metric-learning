from .base_reducer import BaseReducer

class DoNothingReducer(BaseReducer):
    def forward(self, loss_dict, embeddings, labels):
        return loss_dict