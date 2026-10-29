from .dot_product_similarity import DotProductSimilarity


class CosineSimilarity(DotProductSimilarity):
    def __init__(self, *, p: float = 2, power: float = 1) -> None:
        super().__init__(normalize_embeddings=True, p=p, power=power)
