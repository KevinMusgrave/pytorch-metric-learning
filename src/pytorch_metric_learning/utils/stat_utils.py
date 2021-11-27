from . import common_functions as c_f

try:
    import faiss
    import faiss.contrib.torch_utils
except ModuleNotFoundError:
    pass

import numpy as np
import torch


def default_index_init_fn(d):
    res = faiss.StandardGpuResources()
    return faiss.GpuIndexFlatL2(res, d)


class Faiss:
    def __init__(self, index_init_fn=None):
        self.index = None
        self.index_init_fn = (
            default_index_init_fn if index_init_fn is None else index_init_fn
        )

    def save(self, filename):
        faiss.write_index(self.index, filename)

    def load(self, filename):
        self.index = faiss.read_index(filename)

    def init_index(self, d):
        self.index = self.index_init_fn(d)


class FaissKNN(Faiss):
    def __call__(
        self,
        query,
        reference,
        k,
        embeddings_come_from_same_source=False,
    ):
        if embeddings_come_from_same_source:
            k = k + 1
        device = reference.device
        is_cuda = reference.is_cuda
        d = reference.shape[1]
        c_f.LOGGER.info("running k-nn with k=%d" % k)
        c_f.LOGGER.info("embedding dimensionality is %d" % d)
        self.init_index(d)
        distances, indices = try_gpu(
            self.index,
            query.float(),
            reference.float(),
            k,
            is_cuda,
        )
        distances = c_f.to_device(distances, device=device)
        indices = c_f.to_device(indices, device=device)
        if embeddings_come_from_same_source:
            return indices[:, 1:], distances[:, 1:]
        return indices, distances


class FaissKMeans(Faiss):
    # modified from https://raw.githubusercontent.com/facebookresearch/deepcluster/
    def __call__(self, x, nmb_clusters):
        device = x.device
        x = c_f.to_numpy(x).astype(np.float32)
        n_data, d = x.shape
        c_f.LOGGER.info("running k-means clustering with k=%d" % nmb_clusters)
        c_f.LOGGER.info("embedding dimensionality is %d" % d)

        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        index = faiss.IndexFlatL2(d)
        if faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
        # perform the training
        clus.train(x, index)
        _, idxs = index.search(x, 1)

        return torch.tensor([int(n[0]) for n in idxs], dtype=int, device=device)


def add_to_index_and_search(index, test_embeddings, reference_embeddings, k):
    if reference_embeddings is not None:
        index.add(reference_embeddings)
    return index.search(test_embeddings, k)


def convert_to_gpu_index(index):
    if "Gpu" in str(type(index)):
        return index
    return faiss.index_cpu_to_all_gpus(index)


def convert_to_cpu_index(index):
    if "Gpu" not in str(type(index)):
        return index
    return faiss.index_gpu_to_cpu(index)


def try_gpu(index, query, reference, k, is_cuda):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    gpu_condition = is_cuda and gpus_are_available
    if gpu_condition:
        max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
        if k <= max_k_for_gpu:
            gpu_index = convert_to_gpu_index(index)
    try:
        return add_to_index_and_search(gpu_index, query, reference, k)
    except (AttributeError, RuntimeError) as e:
        if gpu_condition:
            c_f.LOGGER.warning(
                f"Using CPU for k-nn search because k = {k} > {max_k_for_gpu}, which is the maximum allowable on GPU."
            )
        cpu_index = convert_to_cpu_index(index)
        return add_to_index_and_search(cpu_index, reference.cpu(), query.cpu(), k)


# modified from https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def run_pca(x, output_dimensionality):
    device = x.device
    x = c_f.to_numpy(x).astype(np.float32)
    mat = faiss.PCAMatrix(x.shape[1], output_dimensionality)
    mat.train(x)
    assert mat.is_trained
    return c_f.to_device(torch.from_numpy(mat.apply_py(x)), device=device)
