import torch

from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsAndDistance


class BaseMiner(ModuleWithRecordsAndDistance):
    def mine(self, embeddings, labels, ref_emb, ref_labels):
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
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = self.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            ref_labels = c_f.to_device(ref_labels, ref_emb)
        else:
            ref_emb, ref_labels = embeddings, labels
        c_f.check_shapes(ref_emb, ref_labels)
        return ref_emb, ref_labels


class BaseTupleMiner(BaseMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(
            list_of_names=["num_pos_pairs", "num_neg_pairs", "num_triplets"],
            is_stat=True,
        )

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
    """

    def __init__(self, output_batch_size, **kwargs):
        super().__init__(**kwargs)
        self.output_batch_size = output_batch_size

    def output_assertion(self, output):
        assert len(output) == self.output_batch_size
