import numpy as np
import faiss

class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query_vec, k=5):
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, k)
        return D, I