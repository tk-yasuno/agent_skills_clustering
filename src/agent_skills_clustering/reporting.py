import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


def top_macro_clusters(mesh: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    return sorted(
        (node for node in mesh["nodes"] if node["type"] == "macro_cluster"),
        key=lambda node: node["skill_count"],
        reverse=True,
    )[:limit]


def build_macro_repository_matrix(
    mesh: dict[str, Any], records: list[dict[str, Any]], *, limit: int = 20
) -> dict[str, Any]:
    macros = top_macro_clusters(mesh, limit=limit)
    macro_ids = [node["id"].split(":", 1)[1] for node in macros]
    repositories = [
        repo
        for repo, _ in Counter(record["repo"] for record in records).most_common(limit)
    ]
    macro_index = {macro_id: index for index, macro_id in enumerate(macro_ids)}
    repo_index = {repo: index for index, repo in enumerate(repositories)}
    values = [[0 for _ in repositories] for _ in macros]
    for edge in mesh["edges"]:
        if edge["type"] != "membership":
            continue
        repo = edge["source"].split(":", 1)[1]
        macro_id = edge["target"].split(":", 1)[1]
        if macro_id in macro_index and repo in repo_index:
            values[macro_index[macro_id]][repo_index[repo]] = edge["weight"]
    return {
        "macro_clusters": [
            {
                "id": macro_id,
                "label": node["label"],
                "skill_count": node["skill_count"],
                "label_keywords": node.get("label_keywords", []),
            }
            for macro_id, node in zip(macro_ids, macros)
        ],
        "repositories": repositories,
        "values": values,
    }


def write_macro_repository_heatmap(
    matrix: dict[str, Any], output_file: Path
) -> None:
    macro_labels = [
        f"{index + 1:02d}. {item['label']} ({item['skill_count']})"
        for index, item in enumerate(matrix["macro_clusters"])
    ]
    figure = go.Figure(
        go.Heatmap(
            z=matrix["values"],
            x=matrix["repositories"],
            y=macro_labels,
            colorscale="YlGnBu",
            hovertemplate=(
                "<b>%{y}</b><br>Repository: %{x}<br>"
                "Skills in macro-cluster: %{z}<extra></extra>"
            ),
            colorbar={"title": "Skill count"},
        )
    )
    figure.update_layout(
        title="Top 20 Macro-Clusters x Top 20 Repositories",
        template="plotly_white",
        xaxis_title="Repository",
        yaxis_title="Macro-cluster",
        height=980,
        margin={"l": 260, "r": 32, "t": 80, "b": 180},
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_file, include_plotlyjs=True, full_html=True)


def write_trend_report(
    mesh: dict[str, Any],
    matrix: dict[str, Any],
    manifest: dict[str, Any],
    output_file: Path,
    *,
    space_url: str,
) -> None:
    top_macros = matrix["macro_clusters"]
    total_skills = manifest["skill_count"]
    summary = mesh.get("summary", {})
    source_cluster_count = summary.get("source_hdbscan_cluster_count", "N/A")
    macro_cluster_count = summary.get("macro_cluster_count", len(top_macros))
    noise_count = summary.get("noise_count", "N/A")
    rows = []
    for index, macro in enumerate(top_macros, start=1):
        row_values = matrix["values"][index - 1]
        repo_index = max(range(len(row_values)), key=row_values.__getitem__)
        top_repo = matrix["repositories"][repo_index] if row_values else "N/A"
        rows.append(
            f"| {index} | {macro['label']} | {macro['skill_count']} | "
            f"{macro['skill_count'] / total_skills:.1%} | `{top_repo}` |"
        )
    top_names = ", ".join(f"`{matrix['repositories'][i]}`" for i in range(min(5, len(matrix["repositories"]))))
    english_post = (
        f"I mapped {total_skills:,} Agent Skills from {manifest['repository_count']} GitHub repositories "
        "into a 2D repository/cluster mesh.\n\n"
        f"The map condenses {source_cluster_count} HDBSCAN groups into "
        f"{macro_cluster_count} macro-clusters and keeps {noise_count} unclustered skills visible. "
        "The heatmap makes the strongest macro-cluster/repository relationships easy to compare.\n\n"
        "The leading themes include document and slide workflows, research and literature, media production, "
        "finance, memory, infrastructure, frontend, and e-commerce. Labels are automated topic hints, not a validated taxonomy.\n\n"
        f"Explore the mesh: {space_url}\n\n"
        "#AgentSkills #OpenSource #AI"
    )
    report = f"""# Agent Skills Trends — September 2026

## Scope

- Repositories: {manifest['repository_count']}
- Valid skills: {manifest['skill_count']:,}
    - Source HDBSCAN clusters: {source_cluster_count}
    - Macro-clusters: {macro_cluster_count}
    - Unclustered skills: {noise_count}
- Heatmap: `macro_repository_heatmap.html`
- Public mesh: {space_url}

## Reader takeaway

Agent Skills are not concentrated in one narrow use case. The top macro-clusters span office documents and slides, research and literature, media production, model/data infrastructure, memory and meeting work, frontend/interface design, e-commerce, finance, and developer operations. The distribution indicates a growing surface area of repeatable agent work rather than a single dominant category.

The heatmap should be read as a relationship map: a bright cell means that a repository contributes many skills to a macro-cluster. It does not mean that the repository owns the category or that the category is a human-validated taxonomy.

The repository examples below are concrete open-source sources for each leading macro-cluster. They are examples, not exclusive memberships.

## Top 20 macro-clusters

| Rank | Macro-cluster | Skills | Share | Example repository |
| ---: | --- | ---: | ---: | --- |
{chr(10).join(rows)}

## Cross-repository signal

The most frequent repositories in the collected sample include {top_names}. The mesh highlights both specialization and breadth: some repositories concentrate on a small number of themes, while others contribute across many macro-clusters. Repository richness is measured with normalized entropy over macro-cluster membership and should be interpreted as breadth of representation, not quality.

## Method and limits

Embeddings use `intfloat/e5-large-v2`; UMAP provides the 2D coordinates; HDBSCAN supplies the source clusters; cosine average-linkage agglomeration creates 100 macro-clusters. Human-readable labels use distinctive TF-IDF terms from `name` and `description` fields where the repository SPDX license is on the reuse allowlist, with repository-name fallback when licensed text is insufficient. Labels are exploratory and require human review before being treated as categories. The sample is fixed to the configured repositories and is not a census of Agent Skills.

## LinkedIn draft

```text
{english_post}
```
"""
    output_file.write_text(report, encoding="utf-8")