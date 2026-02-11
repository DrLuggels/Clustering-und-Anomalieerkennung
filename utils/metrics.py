import numpy as np
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_similarity,
)


def compute_distance_matrices(X: np.ndarray) -> dict[str, np.ndarray]:
    """Compute distance/similarity matrices for all three metrics."""
    return {
        "Euklidisch": euclidean_distances(X),
        "Manhattan": manhattan_distances(X),
        "Cosinus-Ähnlichkeit": cosine_similarity(X),
    }


def pairwise_distances_for_points(X: np.ndarray, idx_a: int, idx_b: int) -> dict[str, float]:
    """Compute all three metrics between two specific points."""
    a = X[idx_a].reshape(1, -1)
    b = X[idx_b].reshape(1, -1)
    return {
        "Euklidisch": float(euclidean_distances(a, b)[0, 0]),
        "Manhattan": float(manhattan_distances(a, b)[0, 0]),
        "Cosinus-Ähnlichkeit": float(cosine_similarity(a, b)[0, 0]),
    }


def distance_comparison_stats(X: np.ndarray) -> dict:
    """Compute average distances and comparison statistics."""
    euc = euclidean_distances(X)
    man = manhattan_distances(X)
    cos = cosine_similarity(X)

    n = X.shape[0]
    mask = np.triu_indices(n, k=1)

    euc_vals = euc[mask]
    man_vals = man[mask]
    cos_vals = cos[mask]

    return {
        "euc_mean": float(np.mean(euc_vals)),
        "euc_std": float(np.std(euc_vals)),
        "man_mean": float(np.mean(man_vals)),
        "man_std": float(np.std(man_vals)),
        "cos_mean": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals)),
        "man_euc_ratio": float(np.mean(man_vals) / (np.mean(euc_vals) + 1e-10)),
    }
