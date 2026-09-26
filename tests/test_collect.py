import pytest

from agent_skills_clustering.collect import collect_repository


class FakeResponse:
    def __init__(self, *, data=None, text=""):
        self.data = data
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/repos/owner/repo"):
            return FakeResponse(
                data={"default_branch": "main", "license": {"spdx_id": "MIT"}}
            )
        if url.endswith("/branches/main"):
            return FakeResponse(data={"commit": {"sha": "abc123"}})
        if url.endswith("/git/trees/abc123"):
            return FakeResponse(
                data={
                    "truncated": False,
                    "tree": [
                        {"type": "blob", "path": "skills/deep/SKILL.md"},
                        {"type": "blob", "path": "README.md"},
                    ],
                }
            )
        if url.endswith("/skills/deep/SKILL.md"):
            return FakeResponse(
                text="---\nname: deep-research\ndescription: Research.\n---\n"
            )
        raise AssertionError(f"Unexpected URL: {url}")


@pytest.mark.parametrize(
    ("pat", "fallback", "expected_token"),
    [
        ("preferred-token", "fallback-token", "preferred-token"),
        (None, "fallback-token", "fallback-token"),
    ],
)
def test_collection_pins_commit_and_keeps_token_off_raw_host(
    monkeypatch, pat, fallback, expected_token
):
    if pat is None:
        monkeypatch.delenv("GH_PAT", raising=False)
    else:
        monkeypatch.setenv("GH_PAT", pat)
    if fallback is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", fallback)
    session = FakeSession()

    records, summary = collect_repository("owner/repo", session=session)

    assert len(records) == 1
    assert records[0].commit_sha == "abc123"
    assert records[0].license_spdx == "MIT"
    assert records[0].raw.startswith("---\nname: deep-research")
    assert summary["skills_collected"] == 1
    api_calls = [call for call in session.calls if "api.github.com" in call[0]]
    raw_calls = [call for call in session.calls if "raw.githubusercontent.com" in call[0]]
    assert len(api_calls) == 3
    assert all(
        call[1]["headers"]["Authorization"] == f"Bearer {expected_token}"
        for call in api_calls
    )
    assert len(raw_calls) == 1
    assert "Authorization" not in raw_calls[0][1].get("headers", {})