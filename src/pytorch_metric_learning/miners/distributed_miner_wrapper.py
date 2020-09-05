import torch
from ..utils import common_functions as c_f

class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner):
        super().__init__()
        self.miner = miner

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        embeddings, labels = c_f.all_gather_embeddings_labels(embeddings, labels)
        if ref_emb is not None:
            ref_emb, ref_labels = all_gather_embeddings_labels(ref_emb, ref_labels)
        return self.miner(embeddings, labels, ref_emb, ref_labels)