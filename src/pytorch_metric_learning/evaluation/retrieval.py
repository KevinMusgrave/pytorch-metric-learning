import importlib
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..distances import BaseDistance, BatchedDistance

FAISS_GPU_MAX_K = 2048
LOGGER = logging.getLogger(__name__)
__all__ = ["CustomKNN", "FaissKMeans", "FaissKNN", "MatchFinder"]


def import_faiss() -> Any:
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
    def __init__(self, distance: BaseDistance, threshold: float | None = None) -> None:
        self.distance = distance
        self.threshold = threshold

    def _threshold(self, threshold: float | None) -> float:
        threshold = self.threshold if threshold is None else threshold
        if threshold is None:
            raise ValueError("A threshold must be supplied to MatchFinder")
        return threshold

    def _compare(self, values: Tensor, threshold: float) -> Tensor:
        if self.distance.is_inverted:
            return values >= threshold
        return values <= threshold

    def get_matching_pairs(
        self,
        query_emb: Tensor,
        ref_emb: Tensor | None = None,
        threshold: float | None = None,
        return_tuples: bool = False,
    ) -> Tensor:
        threshold = self._threshold(threshold)
        ref_emb = query_emb if ref_emb is None else ref_emb
        with torch.no_grad():
            matches = self._compare(self.distance(query_emb, ref_emb), threshold)
        if return_tuples:
            return matches.nonzero()
        return matches

    def is_match(self, x: Tensor, y: Tensor, threshold: float | None = None) -> Tensor:
        threshold = self._threshold(threshold)
        with torch.no_grad():
            return self._compare(self.distance.pairwise(x, y), threshold)


class FaissKNN:
    def __init__(
        self,
        reset_before: bool = True,
        reset_after: bool = True,
        index_init_fn: Callable[[int], Any] | None = None,
        gpus: list[int] | tuple[int, ...] | None = None,
    ) -> None:
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
        query: Tensor,
        k: int,
        reference: Tensor | None = None,
        ref_includes_query: bool = False,
    ) -> tuple[Tensor, Tensor]:
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

    def search(
        self, index: Any, query: Tensor, reference: Tensor | None, k: int
    ) -> tuple[Tensor, Tensor]:
        use_gpu = (
            (query.is_cuda or self.gpus is not None)
            and hasattr(self.faiss, "GpuIndex")
            and self.faiss.get_num_gpus() > 0
        )
        if use_gpu and k <= FAISS_GPU_MAX_K:
            return add_to_index_and_search(
                self.to_gpu_index(index), query, reference, k
            )
        if use_gpu:
            LOGGER.warning(
                "Using CPU for k-nn search because k = %d > %d, "
                "which is the maximum allowable on GPU.",
                k,
                FAISS_GPU_MAX_K,
            )
        return add_to_index_and_search(self.to_cpu_index(index), query, reference, k)

    def to_gpu_index(self, index: Any) -> Any:
        if isinstance(index, self.faiss.GpuIndex):
            return index
        if self.gpus is None:
            return self.faiss.index_cpu_to_all_gpus(index)
        return self.faiss.index_cpu_to_gpus_list(index, gpus=self.gpus)

    def to_cpu_index(self, index: Any) -> Any:
        if hasattr(self.faiss, "GpuIndex") and isinstance(index, self.faiss.GpuIndex):
            return self.faiss.index_gpu_to_cpu(index)
        return index

    def train(self, embeddings: Tensor) -> None:
        self.index = self.index_init_fn(embeddings.shape[1])
        self.add(embeddings)

    def add(self, embeddings: Tensor) -> None:
        if self.index is None:
            raise ValueError("Call train before adding embeddings to FaissKNN")
        self.index.add(torch.as_tensor(embeddings).float().cpu())

    def save(self, filename: str) -> None:
        self.faiss.write_index(self.to_cpu_index(self.index), filename)

    def load(self, filename: str) -> None:
        self.index = self.faiss.read_index(filename)

    def reset(self) -> None:
        self.index = None


class FaissKMeans:
    def __init__(self, **kwargs) -> None:
        self.faiss = import_faiss()
        self.kwargs = kwargs

    def __call__(self, x: Tensor, num_clusters: int) -> Tensor:
        device = x.device
        data = x.detach().float().cpu().numpy()
        d = data.shape[1]
        LOGGER.info("running k-means clustering with k=%d", num_clusters)
        LOGGER.info("embedding dimensionality is %d", d)
        kmeans = self.faiss.Kmeans(d, num_clusters, **self.kwargs)
        kmeans.train(data)
        _, idxs = kmeans.index.search(data, 1)
        return torch.as_tensor(idxs[:, 0], dtype=torch.long, device=device)


def add_to_index_and_search(
    index: Any, query: Tensor, reference: Tensor | None, k: int
) -> tuple[Tensor, Tensor]:
    if reference is not None:
        index.add(reference.float().cpu())
    return index.search(query.float().cpu(), k)


def return_results(
    distances: Tensor, indices: Tensor, ref_includes_query: bool
) -> tuple[Tensor, Tensor]:
    if ref_includes_query:
        self_idx = torch.arange(len(indices), device=indices.device)
        matches_self_idx = indices == self_idx.unsqueeze(1)
        row_has_match = torch.any(matches_self_idx, dim=1)
        matches_self_idx[~row_has_match, -1] = True
        indices = indices[~matches_self_idx].view(indices.shape[0], -1)
        distances = distances[~matches_self_idx].view(distances.shape[0], -1)
    return distances, indices


class CustomKNN:
    def __init__(self, distance: BaseDistance, batch_size: int | None = None) -> None:
        self.distance = distance
        self.batch_size = batch_size

    def __call__(
        self,
        query: Tensor,
        k: int,
        reference: Tensor,
        ref_includes_query: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if ref_includes_query:
            k = k + 1
        largest = self.distance.is_inverted
        if self.batch_size is None:
            distances, indices = torch.topk(
                self.distance(query, reference), k, largest=largest, dim=1
            )
        else:
            distances = torch.empty(
                len(query), k, device=query.device, dtype=query.dtype
            )
            indices = torch.empty(len(query), k, device=query.device, dtype=torch.long)
            batched = BatchedDistance(self.distance, batch_size=self.batch_size)
            for mat, start, end in batched(query, reference):
                distances[start:end], indices[start:end] = torch.topk(
                    mat, k, largest=largest, dim=1
                )
        return return_results(distances, indices, ref_includes_query)
