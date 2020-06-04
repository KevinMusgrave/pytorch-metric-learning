from .base_reducer import BaseReducer
import torch

class MeanReducer(BaseReducer)
    
    def per_element_reducer(self, losses, *_):
        return torch.mean(losses)
    
    def per_pair_reducer(self, losses, *args):
        total_loss = 0
        for sub_loss in losses:
            total_loss += self.per_element_reducer(sub_loss, *args)
        return total_loss     

    def per_triplet_reducer(self, losses, *args):
        return self.per_element_reducer(losses, *args)