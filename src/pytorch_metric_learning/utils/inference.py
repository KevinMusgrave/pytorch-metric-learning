import copy

import numpy as np
import torch

from ..distances import CosineSimilarity
from . import common_functions as c_f


class MatchFinder:
    def __init__(self, distance, threshold=None):
        self.distance = distance
        self.threshold = threshold

    def operate_on_emb(self, input_func, query_emb, ref_emb=None, *args, **kwargs):
        if ref_emb is None:
            ref_emb = query_emb
        return input_func(query_emb, ref_emb, *args, **kwargs)

    # for a batch of queries
    def get_matching_pairs(
        self, query_emb, ref_emb=None, threshold=None, return_tuples=False
    ):
        with torch.no_grad():
            threshold = threshold if threshold is not None else self.threshold
            return self.operate_on_emb(
                self._get_matching_pairs, query_emb, ref_emb, threshold, return_tuples
            )

    def _get_matching_pairs(self, query_emb, ref_emb, threshold, return_tuples):
        mat = self.distance(query_emb, ref_emb)
        matches = mat >= threshold if self.distance.is_inverted else mat <= threshold
        matches = matches.cpu().numpy()
        if return_tuples:
            return list(zip(*np.where(matches)))
        return matches

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        with torch.no_grad():
            dist = self.distance.pairwise_distance(x, y)
            output = (
                dist >= threshold if self.distance.is_inverted else dist <= threshold
            )
            if output.nelement() == 1:
                return output.detach().item()
            return output.cpu().numpy()


class FaissIndexer:
    def __init__(self, index_cls=None):
        import faiss as faiss_module

        self.faiss_module = faiss_module
        self.index_cls = faiss_module.IndexFlatL2 if index_cls is None else index_cls
        self.index = None

    def add_to_index(self, embeddings):
        if self.index is None:
            self.index = self.index_cls(embeddings.shape[1])
        self.index.add(embeddings)

    def get_knn(self, query_batch, k):
        D, I = self.index.search(query_batch, k)
        return I, D

    def save(self, filename):
        self.faiss_module.write_index(self.index, filename)

    def load(self, filename):
        self.index = self.faiss_module.read_index(filename)


class InferenceModel:
    def __init__(
        self,
        trunk,
        embedder=None,
        match_finder=None,
        normalize_embeddings=True,
        indexer=None,
        data_device=None,
        dtype=None,
    ):
        self.trunk = trunk
        self.embedder = c_f.Identity() if embedder is None else embedder
        self.match_finder = (
            MatchFinder(distance=CosineSimilarity(), threshold=0.9)
            if match_finder is None
            else match_finder
        )
        self.indexer = FaissIndexer() if indexer is None else indexer
        self.normalize_embeddings = normalize_embeddings
        self.data_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if data_device is None
            else data_device
        )
        self.dtype = dtype

    def get_embeddings_from_tensor_or_dataset(self, inputs, batch_size):
        inputs = self.process_if_list(inputs)
        embeddings = []
        if isinstance(inputs, (torch.Tensor, list)):
            for i in range(0, len(inputs), batch_size):
                embeddings.append(self.get_embeddings(inputs[i : i + batch_size]))
        elif isinstance(inputs, torch.utils.data.Dataset):
            dataloader = torch.utils.data.DataLoader(inputs, batch_size=batch_size)
            for inp, _ in dataloader:
                embeddings.append(self.get_embeddings(inp))
        else:
            raise TypeError(f"Indexing {type(inputs)} is not supported.")
        return torch.cat(embeddings)

    def train_indexer(self, inputs, batch_size=64):
        self.call_indexer(self.indexer.train_index, inputs, batch_size)

    def add_to_indexer(self, inputs, batch_size=64):
        self.call_indexer(self.indexer.add_to_index, inputs, batch_size)

    def call_indexer(self, func, inputs, batch_size):
        embeddings = self.get_embeddings_from_tensor_or_dataset(inputs, batch_size)
        self.indexer(embeddings.cpu().numpy())

    def get_nearest_neighbors(self, query, k):
        if not self.indexer.index or not self.indexer.index.is_trained:
            raise RuntimeError("Index must be trained by running `train_indexer`")

        query_emb = self.get_embeddings(query)

        indices, distances = self.indexer.search_nn(query_emb.cpu().numpy(), k)
        return indices, distances

    def get_embeddings(self, x):
        x = self.process_if_list(x)
        if isinstance(x, torch.Tensor):
            x = c_f.to_device(x, device=self.data_device, dtype=self.dtype)
        self.trunk.eval()
        self.embedder.eval()
        with torch.no_grad():
            x_emb = self.embedder(self.trunk(x))
        if self.normalize_embeddings:
            x_emb = torch.nn.functional.normalize(x_emb, p=2, dim=1)
        return x_emb

    # for a batch of queries
    def get_matches(self, query, ref=None, threshold=None, return_tuples=False):
        query_emb = self.get_embeddings(query)
        ref_emb = query_emb
        if ref is not None:
            ref_emb = self.get_embeddings(ref)
        return self.match_finder.get_matching_pairs(
            query_emb, ref_emb, threshold, return_tuples
        )

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        x = self.get_embeddings(x)
        y = self.get_embeddings(y)
        return self.match_finder.is_match(x, y, threshold)

    def save_index(self, filename):
        self.indexer.save(filename)

    def load_index(self, filename):
        self.indexer.load(filename)

    def process_if_list(self, x):
        if isinstance(x, list) and all(isinstance(x_, torch.Tensor) for x_ in x):
            return torch.stack(x)
        return x
