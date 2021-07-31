from . import common_functions as c_f

try:
    import faiss
except ModuleNotFoundError:
    c_f.LOGGER.warning(
        """The pytorch-metric-learning testing module requires faiss. You can install the GPU version with the command 'conda install faiss-gpu -c pytorch'
                        or the CPU version with 'conda install faiss-cpu -c pytorch'. Learn more at https://github.com/facebookresearch/faiss/blob/master/INSTALL.md"""
    )

import numpy as np
import torch


class Faiss:
    def __init__(self, index=None, index_type=None, gpus=None):
        self.index = index
        self.index_type = faiss.IndexFlatL2 if index_type is None else index_type
        self.gpus = gpus

    def save(self, filename):
        faiss.write_index(self.index, filename)

    def load(self, filename):
        self.index = faiss.read_index(filename)


class FaissKNN(Faiss):
    def __call__(self, *args, reset_index=True, **kwargs):
        if reset_index:
            self.index = None
        I, D, self.index = get_knn(
            *args,
            faiss_index=self.index,
            faiss_index_type=self.index_type,
            gpus=self.gpus,
            **kwargs,
        )
        return I, D


class FaissKMeans(Faiss):
    def __call__(self, *args, **kwargs):
        return run_kmeans(
            *args, faiss_index_type=self.index_type, gpus=self.gpus, **kwargs
        )


def add_to_index_and_search(index, test_embeddings, reference_embeddings, k):
    if reference_embeddings is not None:
        index.add(reference_embeddings)
    return index.search(test_embeddings, k)


def try_gpu(cpu_index, test_embeddings, reference_embeddings, k, is_cuda, gpus):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    gpu_condition = is_cuda and gpus_are_available
    if gpu_condition:
        max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
        if k <= max_k_for_gpu:
            gpu_index = faiss.index_cpu_to_gpus_list(cpu_index, gpus=gpus)
    try:
        return add_to_index_and_search(
            gpu_index, test_embeddings, reference_embeddings, k
        )
    except (AttributeError, RuntimeError) as e:
        if gpu_condition:
            c_f.LOGGER.warning(
                f"Using CPU for k-nn search because k = {k} > {max_k_for_gpu}, which is the maximum allowable on GPU."
            )
        return add_to_index_and_search(
            cpu_index, test_embeddings, reference_embeddings, k
        )


# modified from https://github.com/facebookresearch/deepcluster
def get_knn(
    test_embeddings,
    k,
    embeddings_come_from_same_source=False,
    reference_embeddings=None,
    faiss_index=None,
    faiss_index_type=None,
    gpus=None,
):
    if embeddings_come_from_same_source:
        k = k + 1
    device = test_embeddings.device
    is_cuda = test_embeddings.is_cuda
    test_embeddings = c_f.to_numpy(test_embeddings).astype(np.float32)

    d = test_embeddings.shape[1]
    c_f.LOGGER.info("running k-nn with k=%d" % k)
    c_f.LOGGER.info("embedding dimensionality is %d" % d)
    if faiss_index is None:
        faiss_index_type = (
            faiss.IndexFlatL2 if faiss_index_type is None else faiss_index_type
        )
        faiss_index = faiss_index_type(d)
    distances, indices = try_gpu(
        faiss_index, test_embeddings, reference_embeddings, k, is_cuda, gpus
    )
    distances = c_f.to_device(torch.from_numpy(distances), device=device)
    indices = c_f.to_device(torch.from_numpy(indices), device=device)
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices, distances, faiss_index


# modified from https://raw.githubusercontent.com/facebookresearch/deepcluster/
def run_kmeans(x, nmb_clusters):
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


# modified from https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def run_pca(x, output_dimensionality):
    device = x.device
    x = c_f.to_numpy(x).astype(np.float32)
    mat = faiss.PCAMatrix(x.shape[1], output_dimensionality)
    mat.train(x)
    assert mat.is_trained
    return c_f.to_device(torch.from_numpy(mat.apply_py(x)), device=device)
