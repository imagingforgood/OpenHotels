"""
Evaluation metrics for hotel retrieval.
"""
import numpy as np
from typing import Dict, List


def compute_recalls(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    retrieved_indices: np.ndarray,
    top_k_list: List[int],
) -> Dict[int, float]:
    """
    Compute Recall@K for hotel-level retrieval.

    A query is correctly matched if the true ``hotel_id`` appears
    among the top-K retrieved gallery images.

    Args:
        query_ids:          (N_q,) array of query hotel IDs (strings).
        gallery_ids:        (N_g,) array of gallery hotel IDs (strings).
        retrieved_indices:  (N_q, max_k) FAISS indices into *gallery_ids*.
        top_k_list:         list of K values, e.g. ``[1, 5, 10, 100]``.

    Returns:
        Dict mapping ``K → recall percentage`` (0–100).
    """
    recalls = {k: 0 for k in top_k_list}
    n_queries = len(query_ids)

    for i in range(n_queries):
        true_id = query_ids[i]
        retrieved_ids = gallery_ids[retrieved_indices[i]]

        for k in top_k_list:
            if true_id in retrieved_ids[:k]:
                recalls[k] += 1

    return {k: (count / n_queries) * 100 for k, count in recalls.items()}


def format_results_table(
    model_name: str,
    split_name: str,
    recalls: Dict[int, float],
) -> str:
    """Format recall results as an ASCII table for terminal output."""
    width = 40
    lines = [
        "=" * width,
        f"  {model_name.upper()} — {split_name}",
        "=" * width,
    ]
    for k, acc in sorted(recalls.items()):
        lines.append(f"  Recall@{k:<4}  {acc:>7.2f}%")
    lines.append("=" * width)
    return "\n".join(lines)
