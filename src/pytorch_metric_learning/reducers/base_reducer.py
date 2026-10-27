import torch

from ..utils import common_functions as c_f


class BaseReducer(torch.nn.Module):
    def forward(self, loss_dict, embeddings, labels):
        assert len(loss_dict) == 1
        loss_name = next(iter(loss_dict))
        loss_info = loss_dict[loss_name]
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        assert_sizes, reduction = {
            "already_reduced": (
                self.assert_sizes_already_reduced,
                self.already_reduced_reduction,
            ),
            "element": (self.assert_sizes_element, self.element_reduction),
            "pos_pair": (self.assert_sizes_pair, self.pos_pair_reduction),
            "neg_pair": (self.assert_sizes_pair, self.neg_pair_reduction),
            "triplet": (self.assert_sizes_triplet, self.triplet_reduction),
        }[reduction_type]
        assert_sizes(losses, loss_indices)
        return reduction(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(
        self, losses, loss_indices, embeddings, labels, **kwargs
    ):
        assert losses.ndim == 0 or len(losses) == 1
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

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)
