import json

from agent_skills_clustering.reporting import (
    build_macro_repository_matrix,
    write_macro_repository_heatmap,
    write_trend_report,
)


def test_reporting_builds_top20_matrix_and_english_trend_report(tmp_path):
    nodes = [
        {
            "id": "cluster:0",
            "type": "macro_cluster",
            "label": "Research & Literature",
            "skill_count": 10,
            "label_keywords": ["research", "literature"],
        },
        {
            "id": "cluster:1",
            "type": "macro_cluster",
            "label": "Video & Audio Production",
            "skill_count": 8,
            "label_keywords": ["video", "audio"],
        },
    ]
    edges = [
        {"source": "repo:owner/research", "target": "cluster:0", "type": "membership", "weight": 7},
        {"source": "repo:owner/video", "target": "cluster:1", "type": "membership", "weight": 6},
    ]
    records = [
        {"repo": "owner/research"},
        {"repo": "owner/research"},
        {"repo": "owner/video"},
    ]
    mesh = {"nodes": nodes, "edges": edges, "summary": {"skill_count": 18}}
    matrix = build_macro_repository_matrix(mesh, records, limit=2)

    assert matrix["repositories"] == ["owner/research", "owner/video"]
    assert matrix["values"] == [[7, 0], [0, 6]]

    heatmap = tmp_path / "heatmap.html"
    report = tmp_path / "TREND_AgentSkiils2026Sept.md"
    write_macro_repository_heatmap(matrix, heatmap)
    write_trend_report(
        mesh,
        matrix,
        {"skill_count": 18, "repository_count": 2},
        report,
        space_url="https://huggingface.co/spaces/example/mesh",
    )

    assert "plotly.js" in heatmap.read_text(encoding="utf-8")
    text = report.read_text(encoding="utf-8")
    assert "Research & Literature" in text
    assert "owner/research" in text
    assert "LinkedIn draft" in text
    assert "https://huggingface.co/spaces/example/mesh" in text
