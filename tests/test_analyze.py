from unittest.mock import patch

import numpy as np
import pytest

from agent_skills_clustering.analyze import analyze_embeddings, cluster_representatives


def test_analysis_clusters_intermediate_umap_output():
    calls = []

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.components = kwargs["n_components"]

        def fit_transform(self, values):
            result = np.ones((len(values), self.components), dtype=np.float32)
            calls.append(result.shape[1])
            return result

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            assert values.shape[1] == 6
            return np.array([0, 0, 1, 1, -1, -1, 0, 1])

    with (
        patch("agent_skills_clustering.analyze.umap.UMAP", FakeUMAP),
        patch("agent_skills_clustering.analyze.hdbscan.HDBSCAN", FakeHDBSCAN),
    ):
        result = analyze_embeddings(
            np.ones((8, 1024), dtype=np.float32),
            n_neighbors=15,
            min_dist=0.1,
            cluster_dimensions=6,
            min_cluster_size=5,
            seed=42,
        )

    assert calls == [6, 2]
    assert result["coordinates"].shape == (8, 2)
    assert result["cluster_count"] == 2
    assert result["noise_count"] == 2


def test_analysis_requires_at_least_three_skills():
    with pytest.raises(ValueError, match="at least 3"):
        analyze_embeddings(
            np.ones((2, 1024), dtype=np.float32),
            n_neighbors=15,
            min_dist=0.1,
            cluster_dimensions=10,
            min_cluster_size=5,
            seed=42,
        )


def test_cluster_representatives_returns_names_nearest_to_centroid():
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = np.array([0, 0, -1])
    result = cluster_representatives(embeddings, labels, ["alpha", "beta", "noise"])

    assert result == {"0": ["alpha", "beta"]}