import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_skills_clustering.mesh import build_skill_mesh
from agent_skills_clustering.models import SkillRecord
from agent_skills_clustering.visualize import write_huggingface_space, write_mesh_visualization

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_mesh(run_directory: Path, *, target_cluster_count: int = 100) -> Path:
    run_directory = run_directory.resolve()
    records_data = json.loads((run_directory / "skills.json").read_text(encoding="utf-8"))
    records = [SkillRecord(**record) for record in records_data]
    embeddings = np.load(run_directory / "embeddings.npy")
    analysis: dict[str, Any] = json.loads(
        (run_directory / "analysis.json").read_text(encoding="utf-8")
    )
    mesh = build_skill_mesh(
        records,
        embeddings,
        np.asarray(analysis["coordinates"], dtype=np.float32),
        np.asarray(analysis["labels"], dtype=np.int32),
        target_cluster_count=target_cluster_count,
    )
    manifest_file = run_directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    represented_repositories = {record.repo for record in records}
    mesh["summary"]["configured_repository_count"] = manifest["repository_count"]
    mesh["summary"]["unrepresented_repositories"] = sorted(
        set(manifest["repositories"]) - represented_repositories
    )

    mesh_file = run_directory / "mesh.json"
    mesh_file.write_text(json.dumps(mesh, ensure_ascii=False, indent=2), encoding="utf-8")
    html_file = run_directory / "mesh.html"
    write_mesh_visualization(mesh, html_file)

    manifest["mesh_results"] = mesh["summary"]
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_huggingface_space(html_file, PROJECT_ROOT / "dist" / "hf-space", manifest)
    return html_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an aggregated repository/cluster skill mesh from an existing run"
    )
    parser.add_argument("run_directory", type=Path, help="A timestamped artifacts run directory")
    parser.add_argument("--clusters", type=int, default=100, help="Target macro-cluster count")
    args = parser.parse_args()
    output = run_mesh(args.run_directory, target_cluster_count=args.clusters)
    print(output)


if __name__ == "__main__":
    main()