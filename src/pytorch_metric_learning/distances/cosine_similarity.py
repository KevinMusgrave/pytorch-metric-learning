from .base_distance import BaseDistance

class CosineSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(small_values_for_large_similarity=True, **kwargs)