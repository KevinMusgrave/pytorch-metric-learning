import torch
from ..utils import common_functions as c_f

class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, embeddings, labels, indices_tuple=None):
        embeddings, labels = c_f.all_gather_embeddings_labels(embeddings, labels)
        return self.loss(embeddings, labels, indices_tuple)