import numpy as np
import torch
from ..utils import common_functions as c_f

from .subcenter_arcface_loss import SubCenterArcFaceLoss
from .arcface_loss import ArcFaceLoss


class DynamicArcFaceLoss(torch.nn.Module):
    """
    Implementation of https://arxiv.org/pdf/2010.05350.pdf
    """

    def __init__(self, n, loss_func=SubCenterArcFaceLoss, lambda0=0.25, a=0.5, b=0.05, **kwargs):
        super().__init__()

        self.lambda0 = lambda0
        self.a = a
        self.b = b

        self.loss_func = loss_func(**kwargs)
        self.n = n.flatten()
        self.init_margins()

    def init_margins(self):
        self.margins = self.a * self.n ** (-self.lambda0) + self.b
    
    def get_batch_margins(self, labels):
        return self.margins[labels]
    
    def set_margins(self, batch_margins):
        self.loss_func.margin = batch_margins
    
    def cast_types(self, tensor, dtype, device):
        return c_f.to_device(tensor, device=device, dtype=dtype)
    
    def forward(self, embeddings, labels):
        batch_margins = self.get_batch_margins(labels)
        dtype, device = embeddings.dtype, embeddings.device
        batch_margins = self.cast_types(batch_margins, dtype, device)
        self.set_margins(batch_margins)
        return self.loss_func(embeddings, labels)

