import torch

from .large_margin_softmax_loss import ArcFaceLoss


class SubCenterArcFaceLoss(ArcFaceLoss):
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