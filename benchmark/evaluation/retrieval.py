"""
FAISS-based retrieval utilities.
"""
import numpy as np
import faiss


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index from L2-normalised embeddings.

    Inner product on unit vectors == cosine similarity, so no extra
    normalisation is needed at search time.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def search_index(
    index: faiss.IndexFlatIP,
    queries: np.ndarray,
    top_k: int,
):
    """
    Search the FAISS index for top-k nearest neighbours.

    Returns:
        distances: (N, top_k)  similarity scores
        indices:   (N, top_k)  gallery row indices
    """
    distances, indices = index.search(queries.astype(np.float32), top_k)
    return distances, indices
