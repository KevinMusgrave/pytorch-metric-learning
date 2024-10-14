import torch


class BaseLossWrapper(torch.nn.Module):
    def __init__(self, loss, **kwargs):
        super().__init__(**kwargs)
        loss_name = type(loss).__name__
        self.check_loss_support(loss_name)

    @staticmethod
    def supported_losses():
        raise NotImplementedError

    @classmethod
    def check_loss_support(self, loss_name):
        raise NotImplementedError
