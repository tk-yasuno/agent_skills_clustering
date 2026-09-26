from agent_skills_clustering.cli import collection_failure_message


def test_collection_failure_message_includes_api_errors_and_auth_hint():
    message = collection_failure_message(
        0,
        [
            {
                "repo": "owner/repo",
                "error": "403 rate limit exceeded",
                "skills_collected": 0,
            }
        ],
    )

    assert "Collected 0 valid skills" in message
    assert "owner/repo: 403 rate limit exceeded" in message
    assert "GH_PAT" in message


def test_collection_failure_message_reports_invalid_skill_files():
    message = collection_failure_message(
        1,
        [
            {
                "repo": "owner/repo",
                "skills_collected": 1,
                "failures": [{"path": "SKILL.md", "error": "missing name"}],
            }
        ],
    )

    assert "Collected 1 valid skills" in message
    assert "Rejected or failed SKILL.md files: 1" in message