from ..utils import common_functions as c_f
from .contrastive_loss import ContrastiveLoss
from ..distances import SNRDistance

class SignalToNoiseRatioContrastiveLoss(ContrastiveLoss):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, SNRDistance)
        
    def get_default_distance(self):
        return SNRDistance()