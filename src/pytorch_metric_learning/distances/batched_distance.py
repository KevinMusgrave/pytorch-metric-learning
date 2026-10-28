import torch


class BatchedDistance(torch.nn.Module):
    def __init__(self, distance, batch_size=32):
        super().__init__()
        self.distance = distance
        self.batch_size = batch_size

    def forward(self, query_emb, ref_emb=None, *, iter_fn):
        ref_emb = ref_emb if ref_emb is not None else query_emb
        n = query_emb.shape[0]
        for s in range(0, n, self.batch_size):
            e = s + self.batch_size
            L = query_emb[s:e]
            mat = self.distance(L, ref_emb)
            iter_fn(mat, s, e)

    @property
    def is_inverted(self):
        return self.distance.is_inverted
