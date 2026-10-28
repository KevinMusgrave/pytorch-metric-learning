import torch

from ..types import IndicesTuple
from ..utils import common_functions as c_f
from ..utils.module_with_distance_and_reducer import ModuleWithDistance


class BaseMiner(ModuleWithDistance):
    def mine(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ref_emb: torch.Tensor,
        ref_labels: torch.Tensor,
    ) -> IndicesTuple:
        raise NotImplementedError

    def validate_output(self, output: IndicesTuple) -> None:
        if not isinstance(output, (tuple, list)) or not all(
            isinstance(index, torch.Tensor) for index in output
        ):
            raise TypeError("Mining must return a tuple or list of index tensors")
        if any(index.ndim != 1 for index in output):
            raise ValueError("Mining indices must be one-dimensional tensors")
        if len(output) == 3:
            if not len(output[0]) == len(output[1]) == len(output[2]):
                raise ValueError(
                    "Triplet mining must return three tensors of equal length"
                )
        elif len(output) == 4:
            if len(output[0]) != len(output[1]):
                raise ValueError(
                    "Positive anchor and partner indices must have equal lengths"
                )
            if len(output[2]) != len(output[3]):
                raise ValueError(
                    "Negative anchor and partner indices must have equal lengths"
                )
        else:
            raise ValueError(
                "Mining must return three triplet or four pair index tensors"
            )

    def prepare_inputs(self, embeddings, labels, ref_emb, ref_labels):
        c_f.check_shapes(embeddings, labels)
        labels = labels.to(device=embeddings.device)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        return labels, ref_emb, ref_labels

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ref_emb: torch.Tensor | None = None,
        ref_labels: torch.Tensor | None = None,
    ) -> IndicesTuple:
        with torch.no_grad():
            labels, ref_emb, ref_labels = self.prepare_inputs(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.validate_output(mining_output)
        return mining_output
