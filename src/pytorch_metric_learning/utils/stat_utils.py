from . import common_functions as c_f

try:
    import faiss
    import faiss.contrib.torch_utils
except ModuleNotFoundError:
    pass

import numpy as np
import torch


def add_to_index_and_search(index, reference_embeddings, test_embeddings, k):
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


def try_gpu(index, reference_embeddings, test_embeddings, k, is_cuda):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    gpu_condition = is_cuda and gpus_are_available
    if gpu_condition:
        max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
        if k <= max_k_for_gpu:
            gpu_index = convert_to_gpu_index(index)
    try:
        return add_to_index_and_search(
            gpu_index, reference_embeddings, test_embeddings, k
        )
    except (AttributeError, RuntimeError) as e:
        if gpu_condition:
            c_f.LOGGER.warning(
                f"Using CPU for k-nn search because k = {k} > {max_k_for_gpu}, which is the maximum allowable on GPU."
            )
        cpu_index = convert_to_cpu_index(index)
        return add_to_index_and_search(
            cpu_index, reference_embeddings.cpu(), test_embeddings.cpu(), k
        )


# modified from https://github.com/facebookresearch/deepcluster
def get_knn(
    reference_embeddings,
    test_embeddings,
    k,
    embeddings_come_from_same_source=False,
    index=None,
):
    if embeddings_come_from_same_source:
        k = k + 1
    device = reference_embeddings.device
    is_cuda = reference_embeddings.is_cuda
    d = reference_embeddings.shape[1]
    c_f.LOGGER.info("running k-nn with k=%d" % k)
    c_f.LOGGER.info("embedding dimensionality is %d" % d)
    if index is None:
        # TODO hope and pray that this issue gets fixed
        # https://github.com/facebookresearch/faiss/issues/1997
        if is_cuda:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)
        else:
            index = faiss.IndexFlatL2(d)
    distances, indices = try_gpu(
        index, reference_embeddings.float(), test_embeddings.float(), k, is_cuda
    )
    distances = c_f.to_device(distances, device=device)
    indices = c_f.to_device(indices, device=device)
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices, distances


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
