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
    noise_display = format(noise_count, ",") if isinstance(noise_count, int) else noise_count
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
        f"{macro_cluster_count} macro-clusters and keeps {noise_display} unclustered skills visible. "
        "The heatmap makes the strongest macro-cluster/repository relationships easy to compare.\n\n"
        "The leading themes include document and slide workflows, research and literature, media production, "
        "finance, memory, infrastructure, frontend, and e-commerce. Labels are automated topic hints, not a validated taxonomy.\n\n"
        f"Explore the mesh: {space_url}\n\n"
        "#AgentSkills #OpenSource #AI"
    )
    report = f"""# Trendo of Agent Skiils 2026 Sept

> Agent Skills landscape: {manifest['repository_count']} repositories, {manifest['skill_count']:,} valid skills, and {macro_cluster_count} macro-clusters.

## What Readers Should Know

- Valid skills: {manifest['skill_count']:,}
- Source HDBSCAN clusters: {source_cluster_count}
- Human-readable macro-clusters: {macro_cluster_count}
    - Unclustered skills: {noise_display}
- Public mesh: {space_url}
- Heatmap: {space_url}/resolve/main/macro_repository_heatmap.html

Agent Skills are being used across a broad range of repeatable tasks rather than one narrow use case. The strongest visible areas are documentation and models, finance and market analysis, frontend and interface design, video and audio production, research and literature, command-line tools, security, cloud infrastructure, office documents, e-commerce, and code review.

Three patterns stand out:

1. **Breadth:** the ecosystem covers both knowledge work and engineering work, from research papers and presentations to deployment, databases, UI, and security.
2. **Specialization and reuse:** some repositories contribute strongly to a narrow theme, while large hubs contribute skills across many macro-clusters.
    3. **A long tail:** {noise_display} skills remain unclustered by HDBSCAN, suggesting many emerging or specialized tasks.

The heatmap is a relationship map: a brighter cell means that a repository contributes more skills to a macro-cluster. It does not mean that the repository owns the category or that the category is a human-validated taxonomy.

## Top 20 Macro-Clusters and Concrete Repository Examples

The table gives one open-source repository example for every leading macro-cluster. These are representative examples, not exclusive memberships.

| Rank | Macro-cluster | Skills | Share | Example repository |
| ---: | --- | ---: | ---: | --- |
{chr(10).join(rows)}

## Current Trend

The most frequent repositories in the collected sample include {top_names}. The leading signal is not a single winning category; it is the expansion of the task surface that agents can repeatedly perform.

The mesh shows a practical distinction between **skill hubs** and **specialists**. Hubs appear across many macro-clusters, while specialist repositories form concentrated bright cells in the heatmap. Repository richness uses normalized entropy as a breadth indicator, not as a quality score.

## Method and Limits

Embeddings use `intfloat/e5-large-v2`; UMAP provides the 2D coordinates; HDBSCAN supplies the source clusters; cosine average-linkage agglomeration creates 100 macro-clusters. Human-readable labels use distinctive TF-IDF terms from `name` and `description` fields where the repository SPDX license is on the reuse allowlist, with repository-name fallback when licensed text is insufficient. Labels are exploratory and require human review before being treated as categories. The sample is fixed to the configured repositories and is not a census of Agent Skills.

## LinkedIn draft

```text
Trendo of Agent Skiils 2026 Sept

{english_post}
```
"""
    output_file.write_text(report, encoding="utf-8")