import torch
from ..utils import common_functions as c_f


class BaseReducer(torch.nn.Module):
    def __init__(self, collect_stats=True):
        super().__init__()
        self.collect_stats = collect_stats

    def forward(self, loss_dict, embeddings, labels):
        c_f.reset_stats(self)
        sub_losses = torch.zeros(len(loss_dict)).to(embeddings.device)
        loss_count = 0
        for self.curr_loss_name, loss_info in loss_dict.items():
            self.add_to_recordable_attributes(name=self.curr_loss_name, is_stat=True, prepend_loss_name=False)
            losses, loss_indices, reduction_type = self.unpack_loss_info(loss_info)
            loss_val = self.reduce_the_loss(losses, loss_indices, reduction_type, embeddings, labels)
            setattr(self, self.curr_loss_name, loss_val)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def unpack_loss_info(self, loss_info):
        return loss_info["losses"], loss_info["indices"], loss_info["reduction_type"]

    def reduce_the_loss(self, losses, loss_indices, reduction_type, embeddings, labels):
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError
    
    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)

    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings*0)

    def input_is_zero_loss(self, losses):
        if (not torch.is_tensor(losses)) and (losses == 0):
            return True
        return False
    
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

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)       

    def assert_sizes_triplet(self, losses, loss_indices): 
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False, optional=False, prepend_loss_name=True):
        if not optional or self.collect_stats: 
            if name is not None:
                if prepend_loss_name:
                    name = self.attribute_namer(name)
                c_f.add_to_recordable_attributes(self, name=name, is_stat=is_stat)
            if list_of_names is not None:
                for name in list_of_names:
                    self.add_to_recordable_attributes(name=name, is_stat=is_stat, prepend_loss_name=prepend_loss_name)

    def get_recordable_attribute(self, name=None, list_of_names=None, prepend_loss_name=True):
        if name is not None:
            if prepend_loss_name:
                name = self.attribute_namer(name)
            return getattr(self, name)
        if list_of_names is not None:
            return [self.get_recordable_attributes(name=name, prepend_loss_name=prepend_loss_name) for name in list_of_names]

    def set_recordable_attribute(self, name, value, prepend_loss_name=True, optional=False):
        if not optional or self.collect_stats: 
            if prepend_loss_name:
                name = self.attribute_namer(name)
            setattr(self, name, value)

    def attribute_namer(self, name):
        return "{}_{}".format(self.curr_loss_name, name)