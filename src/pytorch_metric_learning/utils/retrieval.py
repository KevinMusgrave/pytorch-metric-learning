import importlib
import logging

import torch

from ..distances import BatchedDistance

FAISS_GPU_MAX_K = 2048
LOGGER = logging.getLogger(__name__)


def import_faiss():
    try:
        faiss = importlib.import_module("faiss")
        importlib.import_module("faiss.contrib.torch_utils")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "This retrieval backend requires Faiss. Install the optional "
            "dependency with `pip install pytorch-metric-learning[faiss]`, or "
            "use CustomKNN for PyTorch nearest-neighbor search."
        ) from error
    return faiss


class MatchFinder:
    def __init__(self, distance, threshold=None):
        self.distance = distance
        self.threshold = threshold

    def _threshold(self, threshold):
        threshold = self.threshold if threshold is None else threshold
        if threshold is None:
            raise ValueError("A threshold must be supplied to MatchFinder")
        return threshold

    def _compare(self, values, threshold):
        if self.distance.is_inverted:
            return values >= threshold
        return values <= threshold

    def get_matching_pairs(
        self, query_emb, ref_emb=None, threshold=None, return_tuples=False
    ):
        threshold = self._threshold(threshold)
        ref_emb = query_emb if ref_emb is None else ref_emb
        with torch.no_grad():
            matches = self._compare(self.distance(query_emb, ref_emb), threshold)
        if return_tuples:
            return matches.nonzero()
        return matches

    def is_match(self, x, y, threshold=None):
        threshold = self._threshold(threshold)
        with torch.no_grad():
            x = self.distance.maybe_normalize(x)
            y = self.distance.maybe_normalize(y)
            values = self.distance.pairwise_distance(x, y)
            if self.distance.power != 1:
                values = values**self.distance.power
            return self._compare(values, threshold)


class FaissKNN:
    def __init__(
        self, reset_before=True, reset_after=True, index_init_fn=None, gpus=None
    ):
        self.faiss = import_faiss()
        self.reset()
        self.reset_before = reset_before
        self.reset_after = reset_after
        self.index_init_fn = (
            self.faiss.IndexFlatL2 if index_init_fn is None else index_init_fn
        )
        if gpus is not None:
            if not isinstance(gpus, (list, tuple)):
                raise TypeError("gpus must be a list")
            if len(gpus) < 1:
                raise ValueError("gpus must have length greater than 0")
        self.gpus = gpus

    def __call__(
        self,
        query,
        k,
        reference=None,
        ref_includes_query=False,
    ):
        if ref_includes_query:
            k = k + 1
        device = query.device
        d = query.shape[1]
        LOGGER.info("running k-nn with k=%d", k)
        LOGGER.info("embedding dimensionality is %d", d)
        if self.reset_before:
            self.index = self.index_init_fn(d)
        if self.index is None:
            raise ValueError(
                "self.index is None. It needs to be initialized before being used."
            )
        distances, indices = self.search(self.index, query, reference, k)
        distances = distances.to(device=device)
        indices = indices.to(device=device)
        if self.reset_after:
            self.reset()
        return return_results(distances, indices, ref_includes_query)

    def search(self, index, query, reference, k):
        use_gpu = (query.is_cuda or self.gpus is not None) and (
            self.faiss.get_num_gpus() > 0
        )
        if use_gpu and k <= FAISS_GPU_MAX_K:
            return add_to_index_and_search(
                self.to_gpu_index(index), query, reference, k
            )
        if use_gpu:
            LOGGER.warning(
                f"Using CPU for k-nn search because k = {k} > {FAISS_GPU_MAX_K}, "
                "which is the maximum allowable on GPU."
            )
        return add_to_index_and_search(self.to_cpu_index(index), query, reference, k)

    def to_gpu_index(self, index):
        if isinstance(index, self.faiss.GpuIndex):
            return index
        if self.gpus is None:
            return self.faiss.index_cpu_to_all_gpus(index)
        return self.faiss.index_cpu_to_gpus_list(index, gpus=self.gpus)

    def to_cpu_index(self, index):
        if hasattr(self.faiss, "GpuIndex") and isinstance(index, self.faiss.GpuIndex):
            return self.faiss.index_gpu_to_cpu(index)
        return index

    def train(self, embeddings):
        self.index = self.index_init_fn(embeddings.shape[1])
        self.add(embeddings)

    def add(self, embeddings):
        if self.index is None:
            raise ValueError("Call train before adding embeddings to FaissKNN")
        self.index.add(torch.as_tensor(embeddings).float().cpu())

    def save(self, filename):
        self.faiss.write_index(self.to_cpu_index(self.index), filename)

    def load(self, filename):
        self.index = self.faiss.read_index(filename)

    def reset(self):
        self.index = None


class FaissKMeans:
    def __init__(self, **kwargs):
        self.faiss = import_faiss()
        self.kwargs = kwargs

    def __call__(self, x, nmb_clusters):
        device = x.device
        x = x.detach().float().cpu().numpy()
        d = x.shape[1]
        LOGGER.info("running k-means clustering with k=%d", nmb_clusters)
        LOGGER.info("embedding dimensionality is %d", d)
        kmeans = self.faiss.Kmeans(d, nmb_clusters, **self.kwargs)
        kmeans.train(x)
        _, idxs = kmeans.index.search(x, 1)
        return torch.as_tensor(idxs[:, 0], dtype=torch.long, device=device)


def add_to_index_and_search(index, query, reference, k):
    if reference is not None:
        index.add(reference.float().cpu())
    return index.search(query.float().cpu(), k)


def mask_reshape_knn_idx(x, matches_self_idx):
    return x[~matches_self_idx].view(x.shape[0], -1)


def return_results(distances, indices, ref_includes_query):
    if ref_includes_query:
        self_idx = torch.arange(len(indices), device=indices.device)
        matches_self_idx = indices == self_idx.unsqueeze(1)
        row_has_match = torch.any(matches_self_idx, dim=1)
        if not torch.all(row_has_match):
            matches_self_idx[~row_has_match, -1] = True
        indices = mask_reshape_knn_idx(indices, matches_self_idx)
        distances = mask_reshape_knn_idx(distances, matches_self_idx)
    return distances, indices


def get_topk(distances, indices, k, get_largest):
    def fn(mat, s, e):
        distances[s:e], indices[s:e] = torch.topk(mat, k, largest=get_largest, dim=1)

    return fn


class CustomKNN:
    def __init__(self, distance, batch_size=None):
        if batch_size:
            self.distance = BatchedDistance(distance, batch_size=batch_size)
        else:
            self.distance = distance

    def __call__(self, query, k, reference, ref_includes_query=False):
        if ref_includes_query:
            k = k + 1
        get_largest = self.distance.is_inverted
        if isinstance(self.distance, BatchedDistance):
            distances = torch.zeros(
                len(query), k, device=query.device, dtype=query.dtype
            )
            indices = torch.zeros(len(query), k, device=query.device, dtype=torch.long)
            self.distance(
                query, reference, iter_fn=get_topk(distances, indices, k, get_largest)
            )
        else:
            mat = self.distance(query, reference)
            distances, indices = torch.topk(mat, k, largest=get_largest, dim=1)
        return return_results(distances, indices, ref_includes_query)
