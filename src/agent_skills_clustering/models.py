from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    raw: str
    repo: str
    path: str
    source_url: str
    commit_sha: str
    license_spdx: str | None
    content_sha256: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)