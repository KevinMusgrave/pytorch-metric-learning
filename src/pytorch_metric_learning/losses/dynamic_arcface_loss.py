import numpy as np
import torch

from .subcenter_arcface_loss import SubCenterArcFaceLoss
from .arcface_loss import ArcFaceLoss


class DynamicArcFaceLoss(torch.nn.Module):
    """
    Implementation of https://arxiv.org/pdf/2010.05350.pdf
    """

    def __init__(self, loss_fn, n, lambda0=0.25, a=0.5, b=0.05):
        super().__init__()
        assert isinstance(loss_fn, (ArcFaceLoss, SubCenterArcFaceLoss)), 'Loss function should be Arcface-based'
        self.lambda0 = lambda0
        self.a = a
        self.b = b
        
        self.loss_fn = loss_fn
        self.n = n if len(n.shape) == 2 else n[..., None]
        self.init_margins()

    def init_margins(self):
        self.margins = self.a * self.n ** (-self.lambda0) + self.b
    
    def get_batch_margins(self, labels):
        return self.margins[labels]
    
    def set_margins(self, batch_margins):
        self.loss_fn.margin = batch_margins
    
    def forward(self,  embeddings, labels):
        batch_margins = self.get_batch_margins(labels)
        self.set_margins(batch_margins)
        return self.loss_fn(embeddings, labels)

