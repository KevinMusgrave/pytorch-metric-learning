import torch

class BaseLossWrapper(torch.nn.Module):
    def __init__(self, loss, **kwargs):
        super().__init__(**kwargs)
        if loss is not None:
            # CrossBatchMemory supports loss=None and it's included in the test. how?
            loss_name = type(loss).__name__
            self.check_loss_support(loss_name)

    @staticmethod
    def supported_losses():
        raise NotImplementedError
        
    @classmethod
    def check_loss_support(self, loss_name):
        raise NotImplementedError

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        '''
        Though the implementation may have varying input parameters,
        a LossWrapper must implement this function
        '''
        raise NotImplementedError
    
