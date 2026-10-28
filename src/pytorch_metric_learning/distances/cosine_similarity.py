from .dot_product_similarity import DotProductSimilarity


class CosineSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=True, **kwargs)
