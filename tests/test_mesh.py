import numpy as np

from agent_skills_clustering.mesh import build_skill_mesh
from agent_skills_clustering.models import SkillRecord
from agent_skills_clustering.visualize import write_mesh_visualization


def make_records() -> list[SkillRecord]:
    records = []
    for index in range(12):
        repo = "owner/repo-a" if index < 6 else "owner/repo-b"
        name = f"skill-{index}"
        records.append(
            SkillRecord(
                name=name,
                description=f"Description for {name}",
                raw=f"---\nname: {name}\ndescription: Description for {name}\n---\n",
                repo=repo,
                path=f"skills/{name}/SKILL.md",
                source_url=f"https://github.com/{repo}/blob/abc/skills/{name}/SKILL.md",
                commit_sha="abc",
                license_spdx="MIT",
                content_sha256=f"{index:064x}",
            )
        )
    return records


def test_mesh_merges_hdbscan_clusters_and_preserves_noise():
    records = make_records()
    embeddings = np.vstack(
        [
            np.eye(5, dtype=np.float32)[cluster]
            for cluster in [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1], dtype=np.int32)
    coordinates = np.arange(24, dtype=np.float32).reshape(12, 2)

    mesh = build_skill_mesh(
        records,
        embeddings,
        coordinates,
        labels,
        target_cluster_count=3,
    )

    summary = mesh["summary"]
    node_counts = [
        node["skill_count"]
        for node in mesh["nodes"]
        if node["type"] in {"macro_cluster", "noise"}
    ]
    assert summary["source_hdbscan_cluster_count"] == 5
    assert summary["macro_cluster_count"] == 3
    assert summary["noise_count"] == 2
    assert sum(node_counts) == len(records)
    assert any(node["type"] == "noise" for node in mesh["nodes"])
    assert len([node for node in mesh["nodes"] if node["type"] == "repository"]) == 2
    assert any(edge["type"] == "membership" for edge in mesh["edges"])
    assert any(edge["type"] == "similarity" for edge in mesh["edges"])


def test_mesh_visualization_is_self_contained(tmp_path):
    records = make_records()
    embeddings = np.vstack(
        [np.eye(5, dtype=np.float32)[index % 5] for index in range(len(records))]
    )
    labels = np.array([index % 5 if index < 10 else -1 for index in range(12)])
    coordinates = np.arange(24, dtype=np.float32).reshape(12, 2)
    mesh = build_skill_mesh(
        records,
        embeddings,
        coordinates,
        labels,
        target_cluster_count=3,
    )
    output = tmp_path / "mesh.html"

    write_mesh_visualization(mesh, output)

    document = output.read_text(encoding="utf-8")
    assert "plotly.js" in document
    assert "Macro cluster" in document
    assert "Repository" in document
    assert "Unclustered (HDBSCAN noise)" in document