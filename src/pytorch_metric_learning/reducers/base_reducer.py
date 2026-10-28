import torch

from ..types import LossDict, LossTerm


class BaseReducer(torch.nn.Module):
    def forward(
        self,
        loss_dict: LossDict,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor | LossDict:
        if len(loss_dict) != 1:
            raise ValueError("A reducer requires exactly one loss term")
        loss_name = next(iter(loss_dict))
        loss_info = loss_dict[loss_name]
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        return loss_val

    def unpack_loss_info(self, loss_info: LossTerm):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        reductions = {
            "already_reduced": (
                self.check_sizes_already_reduced,
                self.already_reduced_reduction,
            ),
            "element": (self.check_sizes_element, self.element_reduction),
            "pos_pair": (self.check_sizes_pair, self.pos_pair_reduction),
            "neg_pair": (self.check_sizes_pair, self.neg_pair_reduction),
            "triplet": (self.check_sizes_triplet, self.triplet_reduction),
        }
        if reduction_type not in reductions:
            raise ValueError(f"Unknown reduction_type: {reduction_type}")
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        check_sizes, reduction = reductions[reduction_type]
        check_sizes(losses, loss_indices)
        return reduction(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(
        self, losses, loss_indices, embeddings, labels, **kwargs
    ):
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return self.element_reduction(
            losses, loss_indices[0], embeddings, labels, **kwargs
        )

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return self.element_reduction(
            losses, loss_indices[0], embeddings, labels, **kwargs
        )

    def triplet_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return self.element_reduction(
            losses, loss_indices[0], embeddings, labels, **kwargs
        )

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        return (not torch.is_tensor(losses)) and losses == 0

    def check_sizes_already_reduced(self, losses, loss_indices):
        if not isinstance(losses, torch.Tensor):
            raise TypeError("An already_reduced loss must be a tensor or zero")
        if losses.numel() != 1:
            raise ValueError("An already_reduced loss must contain one value")

    def check_sizes_element(self, losses, loss_indices):
        if not torch.is_tensor(losses):
            raise TypeError("losses must be a tensor")
        if not torch.is_tensor(loss_indices):
            raise TypeError("element indices must be a tensor")
        if losses.ndim == 0 or loss_indices.ndim != 1:
            raise ValueError(
                "Losses need a batch dimension and indices must be one-dimensional"
            )
        if len(losses) != len(loss_indices):
            raise ValueError("losses and element indices must have equal lengths")

    def check_sizes_pair(self, losses, loss_indices):
        if not torch.is_tensor(losses):
            raise TypeError("losses must be a tensor")
        if not isinstance(loss_indices, (list, tuple)):
            raise TypeError(
                "pair and triplet indices must be a tuple or list of tensors"
            )
        if len(loss_indices) != 2:
            raise ValueError("pair indices must contain two tensors")
        if not all(torch.is_tensor(x) for x in loss_indices):
            raise TypeError("every index must be a tensor")
        if losses.ndim == 0 or any(x.ndim != 1 for x in loss_indices):
            raise ValueError(
                "Losses need a batch dimension and indices must be one-dimensional"
            )
        if not len(losses) == len(loss_indices[0]) == len(loss_indices[1]):
            raise ValueError("losses and pair indices must have equal lengths")

    def check_sizes_triplet(self, losses, loss_indices):
        if not torch.is_tensor(losses):
            raise TypeError("losses must be a tensor")
        if not isinstance(loss_indices, (list, tuple)):
            raise TypeError(
                "pair and triplet indices must be a tuple or list of tensors"
            )
        if len(loss_indices) != 3:
            raise ValueError("triplet indices must contain three tensors")
        if not all(isinstance(x, torch.Tensor) for x in loss_indices):
            raise TypeError("every index must be a tensor")
        if losses.ndim == 0 or any(x.ndim != 1 for x in loss_indices):
            raise ValueError(
                "Losses need a batch dimension and indices must be one-dimensional"
            )
        if not all(len(x) == len(losses) for x in loss_indices):
            raise ValueError("losses and triplet indices must have equal lengths")
