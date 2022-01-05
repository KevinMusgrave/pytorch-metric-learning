import torch

from ..utils import common_functions as c_f

class MultipleLossesWrapper(torch.nn.Module):
    def __init__(self, losses, miners=None, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        self.losses = (
            torch.nn.ModuleDict(losses) if self.is_dict else torch.nn.ModuleList(losses)
        )  


        if miners is not None:
            self.assertions_if_not_none(miners, match_all_keys=False)
            self.miners = (
                torch.nn.ModuleDict(miners)
                if self.is_dict
                else torch.nn.ModuleList(miners)
            )
        else:
            self.miners = None

        if weights is not None:
            self.assertions_if_not_none(weights, match_all_keys=True)
            self.weights = weights
        else:
            self.weights = (
                {k: 1 for k in self.losses.keys()}
                if self.is_dict
                else [1] * len(losses)
            )


    def forward(self, embeddings, labels, indices_tuple=None):
        if self.miners:
            assert indices_tuple is None
        total_loss = 0
        iterable = self.losses.items() if self.is_dict else enumerate(self.losses)
        for i, loss_func in iterable:
            curr_indices_tuple = self.get_indices_tuple(
                i, embeddings, labels, indices_tuple
            )
            total_loss += (
                loss_func(embeddings, labels, curr_indices_tuple) * self.weights[i]
            )
        return total_loss

    def get_indices_tuple(self, i, embeddings, labels, indices_tuple):
        if self.miners:
            if (self.is_dict and i in self.miners) or (
                not self.is_dict and self.miners[i] is not None
            ):
                indices_tuple = self.miners[i](embeddings, labels)
        return indices_tuple

    def assertions_if_not_none(self, x, match_all_keys):
        if x is not None:
            if self.is_dict:
                assert isinstance(x, dict)
                if match_all_keys:
                    assert sorted(list(x.keys())) == sorted(list(self.losses.keys()))
                else:
                    assert all(k in self.losses.keys() for k in x.keys())
            else:
                assert c_f.is_list_or_tuple(x)
                assert len(x) == len(self.losses)
