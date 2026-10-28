import torch


class MultipleLosses(torch.nn.Module):
    def __init__(self, losses, miners=None, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        names = list(losses) if self.is_dict else [str(i) for i in range(len(losses))]
        self.losses = torch.nn.ModuleDict(self.by_name(names, losses))
        self.miners = None
        if miners is not None:
            self.check_names(names, miners, match_all_keys=False)
            selected = self.by_name(names, miners)
            self.miners = torch.nn.ModuleDict(
                {k: v for k, v in selected.items() if v is not None}
            )
        self.weights = dict.fromkeys(names, 1)
        if weights is not None:
            self.check_names(names, weights, match_all_keys=True)
            self.weights = self.by_name(names, weights)

    def forward(self, embeddings, labels, indices_tuple=None):
        if self.miners:
            if indices_tuple is not None:
                raise ValueError(
                    "indices_tuple cannot be supplied with configured miners"
                )
        total_loss = 0
        for name, loss_func in self.losses.items():
            curr_indices_tuple = indices_tuple
            if self.miners is not None and name in self.miners:
                curr_indices_tuple = self.miners[name](embeddings, labels)
            loss = loss_func(embeddings, labels, curr_indices_tuple)
            total_loss = total_loss + loss * self.weights[name]
        return total_loss

    def by_name(self, names, values):
        if self.is_dict:
            return dict(values)
        return dict(zip(names, values, strict=True))

    def check_names(self, names, values, match_all_keys):
        if self.is_dict:
            if not isinstance(values, dict):
                raise TypeError(
                    "miners and weights must be dictionaries when losses is a "
                    "dictionary"
                )
            if match_all_keys:
                if sorted(values) != sorted(names):
                    raise ValueError(
                        "weights must have exactly the same keys as losses"
                    )
            else:
                if not all(k in names for k in values):
                    raise ValueError("miner keys must be present in losses")
        else:
            if not isinstance(values, (list, tuple)):
                raise TypeError(
                    "miners and weights must be lists or tuples when losses is a "
                    "sequence"
                )
            if len(values) != len(names):
                raise ValueError(
                    "miners and weights must have the same length as losses"
                )
