import logging

try:
    import faiss
except ModuleNotFoundError:
    logging.warning(
        """The pytorch-metric-learning testing module requires faiss. You can install the GPU version with the command 'conda install faiss-gpu -c pytorch'
                        or the CPU version with 'conda install faiss-cpu -c pytorch'. Learn more at https://github.com/facebookresearch/faiss/blob/master/INSTALL.md"""
    )

import numpy as np
import torch

from . import common_functions as c_f


# modified from https://github.com/facebookresearch/deepcluster
def get_knn(
    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
):
    reference_embeddings = c_f.to_numpy(reference_embeddings).astype(np.float32)
    test_embeddings = c_f.to_numpy(test_embeddings).astype(np.float32)

    d = reference_embeddings.shape[1]
    logging.info("running k-nn with k=%d" % k)
    logging.info("embedding dimensionality is %d" % d)
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    distances, indices = index.search(test_embeddings, k + 1)
    distances = torch.from_numpy(distances)
    indices = torch.from_numpy(indices)
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices[:, :k], distances[:, :k]


# modified from https://raw.githubusercontent.com/facebookresearch/deepcluster/
def run_kmeans(x, nmb_clusters):
    device = x.device
    x = c_f.to_numpy(x).astype(np.float32)
    n_data, d = x.shape
    logging.info("running k-means clustering with k=%d" % nmb_clusters)
    logging.info("embedding dimensionality is %d" % d)

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
    x = c_f.to_numpy(x).astype(np.float32)
    mat = faiss.PCAMatrix(x.shape[1], output_dimensionality)
    mat.train(x)
    assert mat.is_trained
    return c_f.numpy_to_torch(mat.apply_py(x))
