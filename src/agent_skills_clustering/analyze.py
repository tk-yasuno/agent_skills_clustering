from collections import Counter
from typing import Any

import hdbscan
import numpy as np
import umap


def analyze_embeddings(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    cluster_dimensions: int,
    min_cluster_size: int,
    seed: int,
) -> dict[str, Any]:
    count = len(embeddings)
    if count < 3:
        raise ValueError("UMAP visualization requires at least 3 collected skills")

    neighbors = min(n_neighbors, count - 1)
    cluster_components = min(cluster_dimensions, count - 2)
    cluster_space = umap.UMAP(
        n_components=cluster_components,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(embeddings)
    coordinates = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(embeddings)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min(min_cluster_size, count),
        metric="euclidean",
        prediction_data=False,
    ).fit_predict(cluster_space)
    counts = Counter(int(label) for label in labels)

    return {
        "coordinates": np.asarray(coordinates, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int32),
        "cluster_counts": {str(label): size for label, size in sorted(counts.items())},
        "cluster_count": len(counts) - int(-1 in counts),
        "noise_count": counts.get(-1, 0),
        "noise_ratio": round(counts.get(-1, 0) / count, 4),
        "parameters": {
            "n_neighbors": neighbors,
            "min_dist": min_dist,
            "cluster_dimensions": cluster_components,
            "min_cluster_size": min(min_cluster_size, count),
            "seed": seed,
        },
    }


def cluster_representatives(
    embeddings: np.ndarray, labels: np.ndarray, names: list[str], limit: int = 3
) -> dict[str, list[str]]:
    representatives: dict[str, list[str]] = {}
    for label in sorted(set(int(value) for value in labels) - {-1}):
        indexes = np.flatnonzero(labels == label)
        center = embeddings[indexes].mean(axis=0)
        norms = np.linalg.norm(embeddings[indexes], axis=1) * np.linalg.norm(center)
        similarities = embeddings[indexes] @ center / np.maximum(norms, 1e-12)
        ranked = indexes[np.argsort(similarities)[::-1][:limit]]
        representatives[str(label)] = [names[index] for index in ranked]
    return representatives