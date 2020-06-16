from .base_reducer import BaseReducer
import torch


class ClassWeightedReducer(BaseReducer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices, labels)
    
    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices[0], labels)

    # based on anchor label
    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices[0], labels)

    # based on anchor label
    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices[0], labels)

    def element_reduction_helper(self, losses, indices, labels):
        return torch.sum(losses*self.weights[labels[indices]])