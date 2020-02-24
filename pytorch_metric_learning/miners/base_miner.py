#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f

class BaseMiner(torch.nn.Module):
    def __init__(self, normalize_embeddings=True):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Returns: a tuple where each element is an array of indices.
        """
        raise NotImplementedError

    def output_assertion(self, output):
        raise NotImplementedError

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        with torch.no_grad():
            labels = labels.to(embeddings.device)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            ref_emb, ref_labels = self.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            if self.normalize_embeddings:
                ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
            ref_labels = ref_labels.to(ref_emb.device)
        else:
            ref_emb, ref_labels = embeddings, labels
        return ref_emb, ref_labels


    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)
        

class BasePostGradientMiner(BaseMiner):
    """
    A post-gradient miner is used after gradients have already been computed. 
    In other words, the composition of the batch has already been decided, 
    and the miner will find pairs or triplets within the batch that should 
    be used to compute the loss.
    Args:
        normalize_embeddings: type boolean, if True then normalize embeddings
                                to have norm = 1 before mining
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(list_of_names=["num_pos_pairs", "num_neg_pairs", "num_triplets"])


    def output_assertion(self, output):
        """
        Args:
            output: the output of self.mine
        This asserts that the mining function is outputting
        properly formatted indices. The default is to require a tuple representing
        a,p,n indices or a1,p,a2,n indices within a batch of embeddings.
        For example, a tuple of (anchors, positives, negatives) will be
        (torch.tensor, torch.tensor, torch.tensor)
        """
        if len(output) == 3:
            self.num_triplets = len(output[0])
            assert self.num_triplets == len(output[1]) == len(output[2])
        elif len(output) == 4:
            self.num_pos_pairs = len(output[0])
            self.num_neg_pairs = len(output[2])
            assert self.num_pos_pairs == len(output[1])
            assert self.num_neg_pairs == len(output[3])
        else:
            raise BaseException


class BasePreGradientMiner(BaseMiner):
    """
    A pre-gradient miner is used before gradients have been computed.
    The miner finds a subset of the sampled batch for which gradients will
    then need to be computed.
    Args:
        output_batch_size: type int. The size of the subset that the miner
                            will output.
        normalize_embeddings: type boolean, if True then normalize embeddings
                                to have norm = 1 before mining
    """

    def __init__(self, output_batch_size, **kwargs):
        super().__init__(**kwargs)
        self.output_batch_size = output_batch_size

    def output_assertion(self, output):
        assert len(output) == self.output_batch_size