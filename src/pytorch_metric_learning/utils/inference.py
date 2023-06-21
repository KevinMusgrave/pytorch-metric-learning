import numpy as np
import torch

from ..distances import BatchedDistance, CosineSimilarity
from . import common_functions as c_f

try:
    import faiss
    import faiss.contrib.torch_utils
except ModuleNotFoundError:
    pass


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


class InferenceModel:
    def __init__(
        self,
        trunk,
        embedder=None,
        match_finder=None,
        normalize_embeddings=True,
        knn_func=None,
        data_device=None,
        dtype=None,
    ):
        self.trunk = trunk
        self.embedder = torch.nn.Identity() if embedder is None else embedder
        self.match_finder = (
            MatchFinder(distance=CosineSimilarity(), threshold=0.9)
            if match_finder is None
            else match_finder
        )
        self.knn_func = (
            FaissKNN(reset_before=False, reset_after=False)
            if knn_func is None
            else knn_func
        )
        self.normalize_embeddings = normalize_embeddings
        self.data_device = (
            c_f.use_cuda_if_available() if data_device is None else data_device
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

    def train_knn(self, inputs, batch_size=64):
        self.call_knn(self.knn_func.train, inputs, batch_size)

    def add_to_knn(self, inputs, batch_size=64):
        self.call_knn(self.knn_func.add, inputs, batch_size)

    def call_knn(self, func, inputs, batch_size):
        embeddings = self.get_embeddings_from_tensor_or_dataset(inputs, batch_size)
        func(embeddings)

    def get_nearest_neighbors(self, query, k):
        query_emb = self.get_embeddings(query)
        return self.knn_func(query_emb, k)

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

    def save_knn_func(self, filename):
        self.knn_func.save(filename)

    def load_knn_func(self, filename):
        self.knn_func.load(filename)

    def process_if_list(self, x):
        if isinstance(x, list) and all(isinstance(x_, torch.Tensor) for x_ in x):
            return torch.stack(x)
        return x


class FaissKNN:
    def __init__(
        self, reset_before=True, reset_after=True, index_init_fn=None, gpus=None
    ):
        self.reset()
        self.reset_before = reset_before
        self.reset_after = reset_after
        self.index_init_fn = (
            faiss.IndexFlatL2 if index_init_fn is None else index_init_fn
        )
        c_f.check_multiple_gpus(gpus)
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
        is_cuda = query.is_cuda
        d = query.shape[1]
        c_f.LOGGER.info("running k-nn with k=%d" % k)
        c_f.LOGGER.info("embedding dimensionality is %d" % d)
        if self.reset_before:
            self.index = self.index_init_fn(d)
        if self.index is None:
            raise ValueError(
                "self.index is None. It needs to be initialized before being used."
            )
        distances, indices = try_gpu(
            self.index,
            query,
            reference,
            k,
            is_cuda,
            self.gpus,
        )
        distances = c_f.to_device(distances, device=device)
        indices = c_f.to_device(indices, device=device)
        if self.reset_after:
            self.reset()
        return return_results(distances, indices, ref_includes_query)

    def train(self, embeddings):
        self.index = self.index_init_fn(embeddings.shape[1])
        self.add(c_f.numpy_to_torch(embeddings).cpu())

    def add(self, embeddings):
        self.index.add(c_f.numpy_to_torch(embeddings).cpu())

    def save(self, filename):
        faiss.write_index(self.index, filename)

    def load(self, filename):
        self.index = faiss.read_index(filename)

    def reset(self):
        self.index = None


class FaissKMeans:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x, nmb_clusters):
        device = x.device
        x = c_f.to_numpy(x).astype(np.float32)
        n_data, d = x.shape
        c_f.LOGGER.info("running k-means clustering with k=%d" % nmb_clusters)
        c_f.LOGGER.info("embedding dimensionality is %d" % d)

        # faiss implementation of k-means
        kmeans = faiss.Kmeans(d, nmb_clusters, **self.kwargs)
        kmeans.train(x)
        _, idxs = kmeans.index.search(x, 1)
        return torch.tensor([int(n[0]) for n in idxs], dtype=torch.int, device=device)


def add_to_index_and_search(index, query, reference, k):
    indexOnOnlyOneGPU = (not isinstance(index, faiss.IndexShards)) and (isinstance(index, faiss.GpuIndex) or isinstance(index, faiss.GpuIndexShards))    # Issue #491
    if reference is not None:
        index.add(reference.float().cpu())
    return index.search(query.float() if indexOnOnlyOneGPU else query.float().cpu(), k)


def convert_to_gpu_index(index, gpus):
    if "Gpu" in str(type(index)):
        return index
    if gpus is None:
        return faiss.index_cpu_to_all_gpus(index)
    return faiss.index_cpu_to_gpus_list(index, gpus=gpus)


def convert_to_cpu_index(index):
    if "Gpu" not in str(type(index)):
        return index
    return faiss.index_gpu_to_cpu(index)


def try_gpu(index, query, reference, k, is_cuda, gpus):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    gpu_condition = (is_cuda or (gpus is not None)) and gpus_are_available
    max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
    if gpu_condition:
        if k <= max_k_for_gpu:
            gpu_index = convert_to_gpu_index(index, gpus)
    try:
        return add_to_index_and_search(gpu_index, query, reference, k)
    except (AttributeError, RuntimeError):
        if gpu_condition:
            c_f.LOGGER.warning(
                f"Using CPU for k-nn search because k = {k} > {max_k_for_gpu}, which is the maximum allowable on GPU."
            )
        cpu_index = convert_to_cpu_index(index)
        return add_to_index_and_search(cpu_index, query, reference, k)


# modified from https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def run_pca(x, output_dimensionality):
    device = x.device
    x = c_f.to_numpy(x).astype(np.float32)
    mat = faiss.PCAMatrix(x.shape[1], output_dimensionality)
    mat.train(x)
    assert mat.is_trained
    return c_f.to_device(torch.from_numpy(mat.apply_py(x)), device=device)


def mask_reshape_knn_idx(x, matches_self_idx):
    return x[~matches_self_idx].view(x.shape[0], -1)


def return_results(D, I, ref_includes_query):
    if ref_includes_query:
        self_idx = torch.arange(len(I), device=I.device)
        matches_self_idx = I == self_idx.unsqueeze(1)
        row_has_match = torch.any(matches_self_idx, dim=1)
        # If every row has a match, then masking will work
        if not torch.all(row_has_match):
            # For rows that don't contain the self index
            # Remove the Nth value by setting matches_self_idx[N] to True
            matches_self_idx[~row_has_match, -1] = True
        I = mask_reshape_knn_idx(I, matches_self_idx)
        D = mask_reshape_knn_idx(D, matches_self_idx)
    return D, I


def get_topk(distances, indices, k, get_largest):
    def fn(mat, s, e):
        D, I = torch.topk(mat, k, largest=get_largest, dim=1)
        distances[s:e] = D
        indices[s:e] = I

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
            device = query.device
            distances = torch.zeros(len(query), k, device=device)
            indices = torch.zeros(len(query), k, device=device, dtype=torch.long)
            self.distance.iter_fn = get_topk(distances, indices, k, get_largest)
            self.distance(query, reference)
        else:
            mat = self.distance(query, reference)
            distances, indices = torch.topk(mat, k, largest=get_largest, dim=1)
        return return_results(distances, indices, ref_includes_query)
