import numpy as np

from agent_skills_clustering.models import SkillRecord
from agent_skills_clustering.visualize import (
    linkedin_draft,
    write_huggingface_space,
    write_visualization,
)


def test_visualization_is_self_contained_and_hides_unlicensed_descriptions(tmp_path):
    records = [
        SkillRecord(
            name="research",
            description="Reusable description",
            raw="---\nname: research\ndescription: Reusable description\n---\n",
            repo="owner/repo",
            path="research/SKILL.md",
            source_url="https://github.com/owner/repo/blob/abc/research/SKILL.md",
            commit_sha="abc",
            license_spdx="MIT",
            content_sha256="a" * 64,
        ),
        SkillRecord(
            name="private-description",
            description="Do not publish this text",
            raw="---\nname: private-description\ndescription: Do not publish this text\n---\n",
            repo="owner/other",
            path="other/SKILL.md",
            source_url="https://github.com/owner/other/blob/def/other/SKILL.md",
            commit_sha="def",
            license_spdx=None,
            content_sha256="b" * 64,
        ),
    ]
    output = tmp_path / "index.html"

    write_visualization(
        records,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([0, -1]),
        output,
    )
    document = output.read_text(encoding="utf-8")
    decoded_document = document.replace("\\u002f", "/")

    assert "plotly.js" in document
    assert "Reusable description" in document
    assert "Do not publish this text" not in document
    assert "github.com/owner/other" in decoded_document


def test_huggingface_space_includes_cluster_summary(tmp_path):
    html_file = tmp_path / "map.html"
    html_file.write_text("<html>map</html>", encoding="utf-8")
    destination = tmp_path / "space"
    manifest = {
        "skill_count": 10,
        "repository_count": 2,
        "analysis_results": {
            "cluster_count": 3,
            "noise_count": 2,
            "noise_ratio": 0.2,
        },
    }

    write_huggingface_space(html_file, destination, manifest)

    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "HDBSCAN clusters: 3" in readme
    assert "Noise points: 2 (20.0%)" in readme
    assert (destination / "index.html").read_text(encoding="utf-8") == "<html>map</html>"


def test_linkedin_draft_is_concise_english_and_uses_mesh_results():
    manifest = {"skill_count": 11738, "repository_count": 250}
    analysis = {"noise_count": 2946, "noise_ratio": 0.251}
    mesh = {
        "summary": {
            "skill_count": 11738,
            "configured_repository_count": 250,
            "source_hdbscan_cluster_count": 512,
            "macro_cluster_count": 100,
            "noise_count": 2946,
            "labeled_macro_cluster_count": 89,
            "repository_fallback_label_count": 11,
        }
    }

    draft = linkedin_draft(
        manifest,
        analysis,
        mesh=mesh,
        map_url="https://huggingface.co/spaces/yasunotkt/agent-skills-map",
    )

    assert "11,738 Agent Skills" in draft
    assert "100 macro-clusters" in draft
    assert "89 cluster labels" in draft
    assert "not a human-validated taxonomy" in draft
    assert "https://huggingface.co/spaces/yasunotkt/agent-skills-map" in draft
    assert len(draft.split()) < 100