import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from agent_skills_clustering.licensing import REUSABLE_LICENSES
from agent_skills_clustering.models import SkillRecord

def write_visualization(
    records: list[SkillRecord],
    coordinates: np.ndarray,
    labels: np.ndarray,
    output_file: Path,
) -> None:
    descriptions = [
        html.escape(record.description)
        if record.license_spdx in REUSABLE_LICENSES
        else "Description omitted: repository license is not on the reuse allowlist."
        for record in records
    ]
    palette = [
        "#2878B5",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#4E79A7",
    ]
    figure = go.Figure()
    for color_index, label in enumerate(sorted(set(int(value) for value in labels))):
        indexes = np.flatnonzero(labels == label)
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        customdata = [
            [descriptions[index], html.escape(records[index].source_url, quote=True)]
            for index in indexes
        ]
        figure.add_trace(
            go.Scattergl(
                x=coordinates[indexes, 0],
                y=coordinates[indexes, 1],
                mode="markers",
                name=label_name,
                text=[html.escape(records[index].name) for index in indexes],
                customdata=customdata,
                marker={
                    "size": 9,
                    "opacity": 0.82,
                    "color": "#969696" if label == -1 else palette[color_index % len(palette)],
                },
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata[0]}<br>"
                    "<a href='%{customdata[1]}'>GitHub source</a><extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title="Agent Skills Category Map (UMAP + HDBSCAN)",
        template="plotly_white",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="HDBSCAN label",
        margin={"l": 24, "r": 24, "t": 72, "b": 24},
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_file, include_plotlyjs=True, full_html=True)


def write_mesh_visualization(mesh: dict[str, Any], output_file: Path) -> None:
    nodes = mesh["nodes"]
    edges = mesh["edges"]
    positions = {node["id"]: (node["x"], node["y"]) for node in nodes}
    figure = go.Figure()

    for edge_type, name, color, width in (
        ("similarity", "Cluster similarity", "rgba(49, 104, 142, 0.30)", 1.1),
        ("membership", "Repository membership", "rgba(85, 103, 111, 0.18)", 0.8),
    ):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for edge in edges:
            if edge["type"] != edge_type:
                continue
            start = positions.get(edge["source"])
            end = positions.get(edge["target"])
            if start is None or end is None:
                continue
            x_values.extend([start[0], end[0], None])
            y_values.extend([start[1], end[1], None])
        figure.add_trace(
            go.Scattergl(
                x=x_values,
                y=y_values,
                mode="lines",
                name=name,
                line={"color": color, "width": width},
                hoverinfo="skip",
            )
        )

    macro_nodes = [node for node in nodes if node["type"] == "macro_cluster"]
    if macro_nodes:
        customdata = [
            [
                node["skill_count"],
                node["source_count"],
                node["hdbscan_cluster_count"],
                ", ".join(node["label_keywords"]),
                ", ".join(
                    f"{html.escape(item['repo'])} ({item['skill_count']})"
                    for item in node["top_repositories"]
                ),
            ]
            for node in macro_nodes
        ]
        figure.add_trace(
            go.Scattergl(
                x=[node["x"] for node in macro_nodes],
                y=[node["y"] for node in macro_nodes],
                mode="markers",
                name="Macro cluster",
                text=[node["label"] for node in macro_nodes],
                customdata=customdata,
                marker={
                    "size": [
                        min(38, 9 + 4 * np.log1p(node["skill_count"]))
                        for node in macro_nodes
                    ],
                    "color": [node["source_diversity"] for node in macro_nodes],
                    "coloraxis": "coloraxis",
                    "opacity": 0.9,
                    "symbol": "circle",
                    "line": {"color": "#263a42", "width": 1},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>Skills: %{customdata[0]}<br>"
                    "Repositories: %{customdata[1]}<br>"
                    "HDBSCAN clusters merged: %{customdata[2]}<br>"
                    "Topic signals: %{customdata[3]}<br>"
                    "Top repositories: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    noise_nodes = [node for node in nodes if node["type"] == "noise"]
    if noise_nodes:
        figure.add_trace(
            go.Scattergl(
                x=[node["x"] for node in noise_nodes],
                y=[node["y"] for node in noise_nodes],
                mode="markers",
                name="Unclustered skills",
                text=[node["label"] for node in noise_nodes],
                customdata=[[node["skill_count"], node["source_count"]] for node in noise_nodes],
                marker={
                    "size": 13,
                    "color": "#858b8d",
                    "symbol": "diamond",
                    "opacity": 0.85,
                    "line": {"color": "#263a42", "width": 1},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>Skills: %{customdata[0]}<br>"
                    "Repositories: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    repositories = [node for node in nodes if node["type"] == "repository"]
    if repositories:
        customdata = [
            [
                node["skill_count"],
                node["macro_cluster_count"],
                node["macro_cluster_coverage"],
                node["richness"],
                node["noise_count"],
                html.escape(node["source_url"], quote=True),
                ", ".join(
                    f"Macro {item['macro_cluster'] + 1:03d} ({item['skill_count']})"
                    for item in node["top_macro_clusters"]
                ),
            ]
            for node in repositories
        ]
        figure.add_trace(
            go.Scattergl(
                x=[node["x"] for node in repositories],
                y=[node["y"] for node in repositories],
                mode="markers",
                name="Repository",
                text=[html.escape(node["label"]) for node in repositories],
                customdata=customdata,
                marker={
                    "size": [
                        min(25, 8 + 1.5 * np.sqrt(node["skill_count"]))
                        for node in repositories
                    ],
                    "color": [node["richness"] for node in repositories],
                    "coloraxis": "coloraxis",
                    "opacity": 0.92,
                    "symbol": "star",
                    "line": {"color": "#24343a", "width": 0.8},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>Skills: %{customdata[0]}<br>"
                    "Macro-clusters represented: %{customdata[1]}/%{customdata[2]:.0%}<br>"
                    "Distribution richness (normalized entropy): %{customdata[3]:.0%}<br>"
                    "Unclustered skills: %{customdata[4]}<br>"
                    "Top clusters: %{customdata[6]}<br>"
                    "<a href='%{customdata[5]}'>GitHub repository</a>"
                    "<extra></extra>"
                ),
            )
        )

    summary = mesh["summary"]
    figure.update_layout(
        title=(
            "Agent Skills Repository/Cluster Mesh<br>"
            f"<sup>{summary['skill_count']:,} skills | "
            f"{summary['macro_cluster_count']} macro-clusters | "
            f"{summary['repository_count']} repositories</sup>"
        ),
        template="plotly_white",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        hovermode="closest",
        legend_title_text="Mesh node",
        coloraxis={
            "cmin": 0,
            "cmax": 1,
            "colorscale": "Viridis",
            "colorbar": {"title": "Normalized diversity"},
        },
        margin={"l": 28, "r": 28, "t": 86, "b": 28},
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_file, include_plotlyjs=True, full_html=True)


def write_huggingface_space(
    html_file: Path, destination: Path, manifest: dict[str, Any]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(html_file, destination / "index.html")
    mesh_results = manifest.get("mesh_results")
    mesh_summary = ""
    if mesh_results:
        mesh_summary = (
            f"Mesh: {mesh_results['macro_cluster_count']} macro-clusters, "
            f"{mesh_results['repository_count']} repositories, "
            f"mean repository richness "
            f"{mesh_results['mean_repository_richness']:.1%}.\n\n"
            f"Human-readable keyword labels: "
            f"{mesh_results.get('labeled_macro_cluster_count', 0)}; "
            f"repository-name fallbacks: "
            f"{mesh_results.get('repository_fallback_label_count', 0)}.\n\n"
        )
    (destination / "README.md").write_text(
        "---\n"
        "title: Agent Skills Category Map\n"
        "emoji: \U0001f50e\n"
        "colorFrom: blue\n"
        "colorTo: green\n"
        "sdk: static\n"
        "pinned: false\n"
        "---\n\n"
        "# Agent Skills Category Map\n\n"
        "Exploratory visualization of Agent Skills collected from the listed GitHub repositories. "
        "Clusters are algorithmic groupings, not authoritative category labels.\n\n"
        f"Skills: {manifest['skill_count']} | Repositories: {manifest['repository_count']}\n\n"
        f"HDBSCAN clusters: {manifest['analysis_results']['cluster_count']} | "
        f"Noise points: {manifest['analysis_results']['noise_count']} "
        f"({manifest['analysis_results']['noise_ratio']:.1%})\n\n"
        f"{mesh_summary}"
        "Macro-cluster names are automatic keyword summaries from descriptions whose SPDX licenses are on the reuse allowlist. "
        "They are exploratory topic hints, not human-reviewed categories. The displayed topic signals explain each label.\n\n"
        "See the source links in the visualization for provenance.\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def linkedin_draft(
    manifest: dict[str, Any],
    analysis: dict[str, Any],
    *,
    mesh: dict[str, Any] | None = None,
    map_url: str | None = None,
) -> str:
    destination = map_url or "[Add the published Hugging Face Space URL]"
    if mesh is None:
        return (
            "I mapped Agent Skills from GitHub using embeddings, UMAP, and HDBSCAN.\n\n"
            f"This exploratory run covers {manifest['skill_count']:,} skills across "
            f"{manifest['repository_count']} repositories. "
            f"HDBSCAN marked {analysis['noise_count']:,} skills as unclustered.\n\n"
            "Explore the map: " + destination + "\n\n"
            "#AgentSkills #OpenSource #AI"
        )

    summary = mesh["summary"]
    return (
        "I built an interactive 2D mesh of "
        f"{summary['skill_count']:,} Agent Skills from "
        f"{summary['configured_repository_count']} GitHub repositories.\n\n"
        f"The view condenses {summary['source_hdbscan_cluster_count']} HDBSCAN clusters "
        f"into {summary['macro_cluster_count']} macro-clusters while keeping "
        f"{summary['noise_count']:,} unclustered skills visible. Repository links show "
        "how sources span the skill landscape.\n\n"
        f"{summary['labeled_macro_cluster_count']} cluster labels summarize terms from "
        f"license-eligible descriptions; {summary['repository_fallback_label_count']} "
        "fall back to repository names. These are exploratory labels, not a "
        "human-validated taxonomy.\n\n"
        "Explore the mesh: " + destination + "\n\n"
        "#AgentSkills #OpenSource #AI"
    )