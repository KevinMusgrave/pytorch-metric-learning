import torch


class BatchedDistance(torch.nn.Module):
    def __init__(self, distance, iter_fn=None, batch_size=32):
        super().__init__()
        self.distance = distance
        self.iter_fn = iter_fn
        self.batch_size = batch_size

    def forward(self, query_emb, ref_emb=None):
        ref_emb = ref_emb if ref_emb is not None else query_emb
        n = query_emb.shape[0]
        for s in range(0, n, self.batch_size):
            e = s + self.batch_size
            L = query_emb[s:e]
            mat = self.distance(L, ref_emb)
            self.iter_fn(mat, s, e)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.distance, name)
