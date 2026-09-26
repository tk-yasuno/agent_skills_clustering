import numpy as np
from dataclasses import replace

from agent_skills_clustering.mesh import build_skill_mesh
from agent_skills_clustering.mesh import _human_readable_label
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
    assert summary["labeled_macro_cluster_count"] == 0
    assert summary["repository_fallback_label_count"] == 3
    assert sum(node_counts) == len(records)
    assert any(node["type"] == "noise" for node in mesh["nodes"])
    assert len([node for node in mesh["nodes"] if node["type"] == "repository"]) == 2
    assert any(edge["type"] == "membership" for edge in mesh["edges"])
    assert any(edge["type"] == "similarity" for edge in mesh["edges"])


def test_mesh_creates_readable_labels_from_license_allowed_topics():
    records = []
    for index in range(8):
        if index < 4:
            name = f"react-interface-{index}"
            description = "Build React frontend components and responsive user interfaces"
        else:
            name = f"literature-review-{index}"
            description = "Analyze academic research papers and scientific literature"
        records.append(
            SkillRecord(
                name=name,
                description=description,
                raw=f"---\nname: {name}\ndescription: {description}\n---\n",
                repo="owner/research-skills",
                path=f"{name}/SKILL.md",
                source_url=f"https://github.com/owner/research-skills/blob/abc/{name}/SKILL.md",
                commit_sha="abc",
                license_spdx="MIT",
                content_sha256=f"{index:064x}",
            )
        )
    embeddings = np.vstack(
        [np.array([1.0, 0.0], dtype=np.float32)] * 4
        + [np.array([0.0, 1.0], dtype=np.float32)] * 4
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    coordinates = np.arange(16, dtype=np.float32).reshape(8, 2)

    mesh = build_skill_mesh(
        records,
        embeddings,
        coordinates,
        labels,
        target_cluster_count=2,
    )

    cluster_labels = {
        node["id"]: node["label"]
        for node in mesh["nodes"]
        if node["type"] == "macro_cluster"
    }
    assert len(set(cluster_labels.values())) == 2
    assert all(node["label_keywords"] for node in mesh["nodes"] if node["type"] == "macro_cluster")
    assert any("React" in label or "Interface" in label for label in cluster_labels.values())
    assert any("Research" in label or "Literature" in label for label in cluster_labels.values())


def test_mesh_uses_repository_name_when_cluster_text_is_not_reusable():
    records = [replace(record, license_spdx=None) for record in make_records()]
    embeddings = np.vstack(
        [np.array([1.0, 0.0], dtype=np.float32)] * 6
        + [np.array([0.0, 1.0], dtype=np.float32)] * 6
    )
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    coordinates = np.arange(24, dtype=np.float32).reshape(12, 2)

    mesh = build_skill_mesh(
        records,
        embeddings,
        coordinates,
        labels,
        target_cluster_count=2,
    )
    macro_nodes = [node for node in mesh["nodes"] if node["type"] == "macro_cluster"]

    assert {node["label"] for node in macro_nodes} == {
        "Repo A skills",
        "Repo B skills",
    }
    assert all(node["label_method"] == "Repository-name fallback" for node in macro_nodes)
    assert all(not node["label_keywords"] for node in macro_nodes)


def test_human_readable_labels_apply_topic_rules_and_preserve_evidence():
    finance_label, finance_category = _human_readable_label(
        ["berkshire", "excel"], "investing-skills"
    )
    media_label, media_category = _human_readable_label(
        ["video", "hyperframes"], "video-skills"
    )

    assert finance_label == "Finance & Market Analysis · Berkshire"
    assert finance_category == "Finance & Market Analysis"
    assert media_label == "Video & Audio Production · Video"
    assert media_category == "Video & Audio Production"


def test_human_readable_labels_keep_keywords_when_no_topic_rule_matches():
    label, method = _human_readable_label(["signup", "popups", "copy"], "repo-skills")

    assert label == "Signup & Popups"
    assert method == "license-filtered TF-IDF keywords"


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