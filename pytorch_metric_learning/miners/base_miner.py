#! /usr/bin/env python3

import torch


class BaseMiner(torch.nn.Module):
    def __init__(self, normalize_embeddings=True):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

    def mine(self, embeddings, labels):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Returns: a tuple where each element is an array of indices.
        """
        raise NotImplementedError

    def output_assertion(self, output):
        raise NotImplementedError

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        labels = labels.to(embeddings.device)
        with torch.no_grad():
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) 
            mining_output = self.mine(embeddings, labels)
        self.output_assertion(mining_output)
        return mining_output

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
        self.num_pos_pairs = 0
        self.num_neg_pairs = 0
        self.num_triplets = 0
        record_these = ["num_pos_pairs", "num_neg_pairs", "num_triplets"]
        if hasattr(self, "record_these"):
            self.record_these += record_these
        else:
            self.record_these = record_these


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