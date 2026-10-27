import torch

from .base_reducer import BaseReducer


class MeanReducer(BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction(losses, loss_indices, embeddings, labels)

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction(losses, loss_indices, embeddings, labels)

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction(losses, loss_indices, embeddings, labels)
