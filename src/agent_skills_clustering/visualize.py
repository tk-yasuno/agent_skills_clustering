import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from agent_skills_clustering.models import SkillRecord

REUSABLE_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
}


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


def write_huggingface_space(
    html_file: Path, destination: Path, manifest: dict[str, Any]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(html_file, destination / "index.html")
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
        "See the source links in the visualization for provenance.\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def linkedin_draft(manifest: dict[str, Any], analysis: dict[str, Any]) -> str:
    sources = ", ".join(manifest["repositories"])
    return (
        "Agent Skills Category Map (UMAP + HDBSCAN)\n\n"
        "I built an exploratory map of Agent Skills collected from GitHub.\n\n"
        f"- Skills analyzed: {manifest['skill_count']}\n"
        f"- Repositories: {sources}\n"
        f"- Non-noise clusters: {analysis['cluster_count']}\n"
        f"- Noise points: {analysis['noise_count']} "
        f"({analysis['noise_ratio']:.1%})\n\n"
        "Cluster labels are exploratory and should not be read as validated categories.\n"
        "Map: [Add the published Hugging Face Space URL]\n"
    )