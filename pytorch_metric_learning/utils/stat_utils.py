#! /usr/bin/env python3
import faiss
import torch


# modified from https://github.com/facebookresearch/deepcluster
def get_knn(
    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
):
    """
    Finds the k elements in reference_embeddings that are closest to each
    element of test_embeddings.
    Args:
        reference_embeddings: numpy array of size (num_samples, dimensionality).
        test_embeddings: numpy array of size (num_samples2, dimensionality).
        k: int, number of nearest neighbors to find
        embeddings_come_from_same_source: if True, then the nearest neighbor of
                                         each element (which is actually itself)
                                         will be ignored.
    """
    print("running k-nn with k=%d"%k)
    d = reference_embeddings.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    index.add(reference_embeddings)
    _, indices = index.search(test_embeddings, k + 1)
    if embeddings_come_from_same_source:
        return indices[:, 1:]
    return indices[:, :k]


# from https://raw.githubusercontent.com/facebookresearch/deepcluster/
def run_kmeans(x, nmb_clusters):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, idxs = index.search(x, 1)

    return [int(n[0]) for n in idxs]
