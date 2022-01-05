import torch

from .base_loss_wrapper import BaseLossWrapper

class SelfSupervisedLossWrapper(BaseLossWrapper):
    '''
    Issue #411:
    
    A common use case is to have embeddings and ref_emb be augmented versions of each other. 
    For most losses right now you have to create labels to indicate 
    which embeddings correspond with which ref_emb. 
    A wrapper that does this for the user would be nice.

        loss_fn = SelfSupervisedLossWrapper(TripletMarginLoss())
        loss = loss_fn(embeddings, ref_emb1, ref_emb2, ...)
    
    where ref_embk = kth augmentation of embeddings.
    '''
    def __init__(
        self,
        loss,
        **kwargs
    ):
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
    
    @staticmethod
    def supported_losses():
        '''
        losses.LiftedStructureLoss leads to:
            File "/workspace/chanwookim/pytorch-metric-learning/tests/wrappers/test_self_supervised_loss_wrapper.py", line 66, in test_ssl_wrapper_all
                real_losses = self.run_all_loss_fns(embeddings, labels, ref_emb, labels)
            File "/workspace/chanwookim/pytorch-metric-learning/tests/wrappers/test_self_supervised_loss_wrapper.py", line 85, in run_all_loss_fns
                ref_labels=ref_labels
            File "/opt/conda/envs/ACW/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
                return forward_call(*input, **kwargs)
            File "src/pytorch_metric_learning/losses/base_metric_loss_function.py", line 38, in forward
                embeddings, labels, indices_tuple, ref_emb, ref_labels
            File "src/pytorch_metric_learning/losses/generic_pair_loss.py", line 19, in compute_loss
                return self.loss_method(mat, indices_tuple)
            File "src/pytorch_metric_learning/losses/generic_pair_loss.py", line 38, in pair_based_loss
                return self._compute_loss(pos_pair, neg_pair, indices_tuple)
            File "src/pytorch_metric_learning/losses/lifted_structure_loss.py", line 19, in _compute_loss
                dtype = pos_pairs.dtype
            AttributeError: 'list' object has no attribute 'dtype'
        '''
        
        return [
            "AngularLoss",
            "CircleLoss",
            "ContrastiveLoss",
            "GeneralizedLiftedStructureLoss",
            "IntraPairVarianceLoss",
            # "LiftedStructureLoss",
            "MultiSimilarityLoss",
            "NTXentLoss",
            "SignalToNoiseRatioContrastiveLoss",
            "SupConLoss",
            "TripletMarginLoss",

            "NCALoss",
            "TupletMarginLoss"
        ]
    
    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise Exception(f"SelfSupervisedLossWrapper not supported for {loss_name}") 

    def compute_loss(self, embeddings, ref_emb, *args):
        '''
        embeddings: representations of the original set of inputs
        ref_emb:    representations of an augmentation of the inputs.   
        *args:      variable length argument list, where each argument
                    is an additional representation of an augmented version of the input.  
                    i.e. ref_emb2, ref_emb3, ...
        '''
        embeddings_size = embeddings.size()
        ref_emb_size = ref_emb.size()

        embeddings_type = embeddings.type()
        ref_emb_type = ref_emb.type()

        embeddings_device = embeddings.get_device()
        ref_emb_device = ref_emb.get_device()

        if embeddings_size != ref_emb_size:
            raise Exception(f"Input parameters embeddings and ref_emb must be of the same size. Found '{embeddings_size}' and '{ref_emb_size}' instead.")
        if embeddings_type != ref_emb_type:
            raise Exception(f"Input parameters embeddings and ref_emb must be of the type. Found '{embeddings_type}' and '{ref_emb_type}' instead.")
        if embeddings_device != ref_emb_device:
            raise Exception(f"Input parameters embeddings and ref_emb must be on the same device. Found '{embeddings_device}' and '{ref_emb_device}' instead.")

        batch_size = embeddings_size[0]
        labels = torch.arange(batch_size).type(embeddings_type).to(embeddings_device)
        return self.loss(
            embeddings=embeddings, 
            labels=labels, 
            ref_emb=ref_emb,
            ref_labels=labels
        )

    def forward(self, embeddings, ref_emb, *args):
        return self.compute_loss(embeddings, ref_emb, *args)

    @staticmethod
    def get_blacklist():
        return [
            # Not loss functions
            "BaseMetricLossFunction",
            "EmbeddingRegularizerMixin",
            "GenericPairLoss",
            "MultipleLosses",

            # Requires num_classes at initialization
            "ArcFaceLoss",
            "LargeMarginSoftmaxLoss",  
            "NormalizedSoftmaxLoss",
            "ProxyAnchorLoss",
            "ProxyNCALoss",
            "SignalToNoiseRatioContrastiveLoss",
            "SoftTripleLoss",
            "SphereFaceLoss",
            
            # Requires "num_pairs" at initialization
            "NPairsLoss",

            # Not applicable
            "FastAPLoss",
            "CentroidTripletLoss",
            "VICRegLoss",
            "AngularLoss",
            "LiftedStructureLoss",
            "NCALoss",
            "TupletMarginLoss"
        ]

