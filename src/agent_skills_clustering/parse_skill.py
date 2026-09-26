import hashlib

import frontmatter
import yaml

from agent_skills_clustering.models import SkillRecord


def parse_skill(
    raw: str,
    *,
    repo: str,
    path: str,
    commit_sha: str,
    license_spdx: str | None,
) -> SkillRecord:
    try:
        post = frontmatter.loads(raw)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid SKILL.md frontmatter YAML: {error}") from error
    name = post.metadata.get("name")
    description = post.metadata.get("description")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("SKILL.md frontmatter must contain a non-empty string 'name'")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "SKILL.md frontmatter must contain a non-empty string 'description'"
        )

    source_url = f"https://github.com/{repo}/blob/{commit_sha}/{path}"
    return SkillRecord(
        name=name.strip(),
        description=description.strip(),
        raw=raw,
        repo=repo,
        path=path,
        source_url=source_url,
        commit_sha=commit_sha,
        license_spdx=license_spdx,
        content_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )