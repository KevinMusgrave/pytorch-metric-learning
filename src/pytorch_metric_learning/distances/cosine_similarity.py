from .dot_product_similarity import DotProductSimilarity


class CosineSimilarity(DotProductSimilarity):
    def __init__(self) -> None:
        super().__init__(normalize_embeddings=True)
