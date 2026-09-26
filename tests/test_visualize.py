import numpy as np

from agent_skills_clustering.models import SkillRecord
from agent_skills_clustering.visualize import (
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