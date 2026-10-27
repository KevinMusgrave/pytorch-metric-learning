import torch

from ..utils import common_functions as c_f
from ..utils.module_with_reducer import ModuleWithDistance


class BaseMiner(ModuleWithDistance):
    def mine(self, embeddings, labels, ref_emb, ref_labels):
        raise NotImplementedError

    def output_assertion(self, output):
        if len(output) == 3:
            assert len(output[0]) == len(output[1]) == len(output[2])
        elif len(output) == 4:
            assert len(output[0]) == len(output[1])
            assert len(output[2]) == len(output[3])
        else:
            raise TypeError

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = c_f.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output
