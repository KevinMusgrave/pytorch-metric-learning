import torch
import torch.nn.functional as F
import math
from .large_margin_softmax_loss import ArcFaceLoss


class SubCenterArcFaceLoss(losses.ArcFaceLoss):
    """
    Implementation of https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf
    """

    def __init__(self, *args, margin=28.6, scale=64, sub_centers=3, **kwargs):
        num_classes, embedding_size = args
        super().__init__(num_classes * sub_centers, embedding_size, margin=margin, scale=scale, **kwargs)
        self.sub_centers = sub_centers
        self.num_classes = num_classes

    def get_cosine(self, embeddings):
        cosine = self.distance(embeddings, self.W.t())
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = cosine.max(axis=2)
        return cosine

    def get_outliers(self, embeddings, labels, threshold=75, return_dominant_centers=True, normalize=True):
        self.eval()
        if len(labels.shape) > 1:
            labels = labels.flatten()
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        device = self.W.device
        cos_thresh = math.cos(math.pi * threshold / 180.)
        outliers = []
        dominant_centers = torch.Tensor(embeddings.shape[1], self.num_classes)
        with torch.set_grad_enabled(False):
            for label in range(self.num_classes):
                target_samples = labels == label
                target_indeces = target_samples.nonzero()
                target_embeddings = embeddings[target_samples]

                sub_centers = self.W[:, label * self.sub_centers:(label + 1) * self.sub_centers]
                sub_centers = F.normalize(sub_centers, p=2, dim=0)
                distances = torch.mm(target_embeddings, sub_centers)
                max_sub_center_idxs = torch.argmax(distances, axis=1)
                max_sub_center_count = torch.bincount(max_sub_center_idxs)
                dominant_idx = torch.argmax(max_sub_center_count)
                dominant_center = sub_centers[:, dominant_idx]
                dominant_centers[:, label] = dominant_center
                
                dominant_dist = distances[:, dominant_idx]
                drop_dists = dominant_dist < cos_thresh
                drop_idxs = target_indeces[drop_dists] 
                outliers.extend(drop_idxs.detach().tolist())
        outliers = torch.tensor(outliers).flatten()
        return outliers if not return_dominant_centers else outliers, dominant_centers
    