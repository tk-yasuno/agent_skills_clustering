import pytest

from agent_skills_clustering.parse_skill import parse_skill


def test_parse_multiline_frontmatter_and_source_metadata():
    raw = """---
name: deep-research
description: >-
  Use when the user needs research
  across multiple sources.
---
Body text is not part of the feature.
"""

    record = parse_skill(
        raw,
        repo="example/skills",
        path="research/SKILL.md",
        commit_sha="abc123",
        license_spdx="MIT",
    )

    assert record.name == "deep-research"
    assert record.description == "Use when the user needs research across multiple sources."
    assert record.raw == raw
    assert record.to_dict()["raw"] == raw
    assert record.source_url.endswith("/blob/abc123/research/SKILL.md")
    assert len(record.content_sha256) == 64


@pytest.mark.parametrize(
    "raw",
    [
        "---\ndescription: A skill\n---\n",
        "---\nname: skill\n---\n",
        "---\nname: 4\ndescription: A skill\n---\n",
        "---\nname: [unterminated\ndescription: A skill\n---\n",
    ],
)
def test_rejects_missing_or_invalid_required_fields(raw: str):
    with pytest.raises(ValueError):
        parse_skill(
            raw,
            repo="example/skills",
            path="SKILL.md",
            commit_sha="abc123",
            license_spdx=None,
        )