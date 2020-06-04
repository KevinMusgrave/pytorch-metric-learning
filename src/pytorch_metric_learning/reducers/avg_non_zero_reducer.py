from .base_reducer import BaseReducer


class AvgNonZeroReducer(BaseReducer):
    def per_element_reducer(self, losses, *_):
        num_non_zero = len((losses > 0).nonzero())
        if num_non_zero >= 1:
            return torch.sum(losses) / num_non_zero
        else:
            output = torch.mean(losses)
            assert output == 0 # mean must be zero, otherwise something is wrong
            return output 
    
    def per_pair_reducer(self, losses, *args):
        total_loss = 0
        for sub_loss in losses:
            total_loss += self.per_element_reducer(sub_loss, *args)
        return total_loss 
        
    def per_triplet_reducer(self, losses, *args):
        return self.per_element_reducer(losses, *args)