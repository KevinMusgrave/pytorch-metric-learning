import torch
from torch import Tensor

from ..utils._common import LOW_PRECISION, accumulation_dtype, input_dtype
from .base_distance import BaseDistance


def _euclidean_matrix(
    query: Tensor, reference: Tensor, normalized: bool, input_epsilon: float
) -> Tensor:
    mat = torch.cdist(query, reference)
    with torch.no_grad():
        scale = (
            2 * (1 + input_epsilon) ** 2
            if normalized
            else query.square().sum(-1, keepdim=True)
            + reference.square().sum(-1).unsqueeze(0)
        )
        bound = 8 * query.shape[-1] * torch.finfo(query.dtype).eps / input_epsilon
        risky = (mat.square() <= bound * scale) | ~torch.isfinite(mat)
        if query is reference:
            risky.fill_diagonal_(False)
    if risky.any():
        return torch.cdist(
            query, reference, compute_mode="donot_use_mm_for_euclid_dist"
        )
    if query is reference:
        mat = mat.diagonal_scatter(torch.zeros_like(mat.diagonal()))
    return mat


class LpDistance(BaseDistance):
    def compute_mat(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        if input_dtype(query_emb) in LOW_PRECISION:
            query = query_emb.float()
            reference = query if query_emb is ref_emb else ref_emb.float()
            if (
                self.p == 2
                and max(query.shape[0], reference.shape[0]) > 25
                and not torch.compiler.is_compiling()
                and torch.get_float32_matmul_precision() == "highest"
                and not (
                    query.device.type == "cuda"
                    and torch.cuda.is_current_stream_capturing()
                )
            ):
                mat = _euclidean_matrix(
                    query,
                    reference,
                    self.normalize_embeddings,
                    torch.finfo(input_dtype(query_emb)).eps,
                )
            else:
                mat = torch.cdist(
                    query,
                    reference,
                    p=self.p,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
            if self.normalize_embeddings:
                return mat.to(query_emb.dtype)
            return mat
        return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        dtype = accumulation_dtype(query_emb.dtype)
        difference = query_emb.to(dtype) - ref_emb.to(dtype)
        distances = torch.linalg.vector_norm(difference, ord=self.p, dim=-1)
        if self.normalize_embeddings:
            return distances.to(query_emb.dtype)
        return distances

    def prepare_input(self, embeddings: Tensor) -> Tensor:
        return self.maybe_normalize(embeddings)
