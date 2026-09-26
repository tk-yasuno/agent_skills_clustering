import os
import time
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent_skills_clustering.models import SkillRecord
from agent_skills_clustering.parse_skill import parse_skill

GITHUB_API = "https://api.github.com"


def github_token() -> str | None:
    return os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "User-Agent": "agent-skills-clustering/0.2.2",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    retry = Retry(
        total=4,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def collect_repository(
    repo: str, *, session: requests.Session | None = None
) -> tuple[list[SkillRecord], dict[str, object]]:
    client = session or make_session()
    token = github_token()
    api_headers = {"Authorization": f"Bearer {token}"} if token else {}
    repository = client.get(
        f"{GITHUB_API}/repos/{repo}", headers=api_headers, timeout=30
    )
    repository.raise_for_status()
    repository_data = repository.json()
    branch = repository_data["default_branch"]

    branch_response = client.get(
        f"{GITHUB_API}/repos/{repo}/branches/{quote(branch, safe='')}",
        headers=api_headers,
        timeout=30,
    )
    branch_response.raise_for_status()
    commit_sha = branch_response.json()["commit"]["sha"]
    license_data = repository_data.get("license") or {}
    license_spdx = license_data.get("spdx_id")

    tree_response = client.get(
        f"{GITHUB_API}/repos/{repo}/git/trees/{commit_sha}",
        params={"recursive": "1"},
        headers=api_headers,
        timeout=60,
    )
    tree_response.raise_for_status()
    tree_data = tree_response.json()
    if tree_data.get("truncated"):
        raise RuntimeError(f"GitHub returned a truncated tree for {repo}")

    skill_paths = sorted(
        item["path"]
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").rsplit("/", 1)[-1].lower() == "skill.md"
    )
    records: list[SkillRecord] = []
    failures: list[dict[str, str]] = []

    for path in skill_paths:
        raw_url = (
            f"https://raw.githubusercontent.com/{repo}/{commit_sha}/"
            f"{quote(path, safe='/')}"
        )
        try:
            response = client.get(raw_url, timeout=30)
            response.raise_for_status()
            records.append(
                parse_skill(
                    response.text,
                    repo=repo,
                    path=path,
                    commit_sha=commit_sha,
                    license_spdx=license_spdx,
                )
            )
        except (requests.RequestException, ValueError) as error:
            failures.append({"path": path, "error": str(error)})
        time.sleep(0.05)

    metadata: dict[str, object] = {
        "repo": repo,
        "default_branch": branch,
        "commit_sha": commit_sha,
        "license_spdx": license_spdx,
        "skill_paths_found": len(skill_paths),
        "skills_collected": len(records),
        "failures": failures,
    }
    return records, metadata


def collect_repositories(
    repositories: list[str], *, session: requests.Session | None = None
) -> tuple[list[SkillRecord], list[dict[str, object]]]:
    all_records: list[SkillRecord] = []
    summaries: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for repo in repositories:
        try:
            records, summary = collect_repository(repo, session=session)
        except (requests.RequestException, RuntimeError, KeyError) as error:
            records = []
            summary = {
                "repo": repo,
                "skills_collected": 0,
                "error": str(error),
            }
        summaries.append(summary)
        for record in records:
            key = (record.repo.casefold(), record.path.casefold())
            if key not in seen:
                seen.add(key)
                all_records.append(record)
    return all_records, summaries