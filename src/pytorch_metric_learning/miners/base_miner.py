#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsAndDistance

class BaseMiner(ModuleWithRecordsAndDistance):
    def __init__(self, normalize_embeddings=True, **kwargs):
        super().__init__(**kwargs)
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
        self.reset_stats()
        with torch.no_grad():
            c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
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
        c_f.assert_embeddings_and_labels_are_same_size(ref_emb, ref_labels)
        return ref_emb, ref_labels


class BaseTupleMiner(BaseMiner):
    """
    Args:
        normalize_embeddings: type boolean, if True then normalize embeddings
                                to have norm = 1 before mining
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(list_of_names=["num_pos_pairs", "num_neg_pairs", "num_triplets"], is_stat=True)

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


class BaseSubsetBatchMiner(BaseMiner):
    """
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