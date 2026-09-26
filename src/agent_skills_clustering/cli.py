import argparse
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agent_skills_clustering import __version__
from agent_skills_clustering.analyze import analyze_embeddings, cluster_representatives
from agent_skills_clustering.collect import collect_repositories
from agent_skills_clustering.embed import cuda_runtime, embed_texts
from agent_skills_clustering.visualize import (
    linkedin_draft,
    write_huggingface_space,
    write_visualization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("agent_skills_clustering")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def collection_failure_message(
    collected_count: int, source_summaries: list[dict[str, Any]]
) -> str:
    errors = [
        f"{summary.get('repo', 'unknown repository')}: {summary['error']}"
        for summary in source_summaries
        if summary.get("error")
    ]
    rejected_count = sum(
        len(summary.get("failures", [])) for summary in source_summaries
    )
    message = f"Collected {collected_count} valid skills; at least 3 are required."
    if errors:
        shown_errors = errors[:5]
        if len(errors) > len(shown_errors):
            shown_errors.append(f"and {len(errors) - len(shown_errors)} more repository errors")
        message += " Repository errors: " + " | ".join(shown_errors) + "."
        message += " Check GH_PAT (preferred) or GITHUB_TOKEN and GitHub API quota."
    if rejected_count:
        message += f" Rejected or failed SKILL.md files: {rejected_count}."
    return message


def run_pipeline(config_path: Path, *, use_cpu: bool = False) -> Path:
    config = load_yaml(config_path)
    source_config = load_yaml(PROJECT_ROOT / config["sources_file"])
    repositories = source_config.get("repositories", [])
    if not repositories:
        raise ValueError("No repositories configured in the sources file")

    if not use_cpu:
        runtime = cuda_runtime()
        LOGGER.info("Using %s (%s GiB)", runtime["gpu_name"], runtime["gpu_memory_gib"])

    records, source_summaries = collect_repositories(repositories)
    if len(records) < 3:
        raise RuntimeError(collection_failure_message(len(records), source_summaries))

    embedding_config = config["embedding"]
    features = [f"{record.name}. {record.description}" for record in records]
    embeddings, runtime = embed_texts(
        features,
        model_name=embedding_config["model"],
        batch_size=int(embedding_config["batch_size"]),
        seed=int(config["seed"]),
        use_cpu=use_cpu,
    )
    analysis_config = config["analysis"]
    analysis = analyze_embeddings(
        embeddings,
        n_neighbors=int(analysis_config["n_neighbors"]),
        min_dist=float(analysis_config["min_dist"]),
        cluster_dimensions=int(analysis_config["cluster_dimensions"]),
        min_cluster_size=int(analysis_config["min_cluster_size"]),
        seed=int(config["seed"]),
    )
    analysis["representatives"] = cluster_representatives(
        embeddings, analysis["labels"], [record.name for record in records]
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_root = PROJECT_ROOT / config["artifacts_dir"] / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    (artifact_root / "skills.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(artifact_root / "embeddings.npy", embeddings)
    (artifact_root / "analysis.json").write_text(
        json.dumps(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in analysis.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": __version__,
        "skill_count": len(records),
        "repository_count": len(repositories),
        "repositories": repositories,
        "sources": source_summaries,
        "embedding": {**runtime, "seed": int(config["seed"])},
        "analysis": analysis["parameters"],
        "analysis_results": {
            "cluster_count": analysis["cluster_count"],
            "cluster_counts": analysis["cluster_counts"],
            "noise_count": analysis["noise_count"],
            "noise_ratio": analysis["noise_ratio"],
            "representatives": analysis["representatives"],
            "cluster_labels_are_exploratory": True,
        },
        "versions": {
            name: package_version(name)
            for name in ("numpy", "umap-learn", "hdbscan", "plotly", "torch")
        },
    }
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_file = artifact_root / "index.html"
    write_visualization(
        records, analysis["coordinates"], analysis["labels"], html_file
    )
    write_huggingface_space(
        html_file, PROJECT_ROOT / "dist" / "hf-space", manifest
    )
    (artifact_root / "linkedin_draft.md").write_text(
        linkedin_draft(manifest, analysis), encoding="utf-8"
    )
    LOGGER.info("Created map and run artifacts in %s", artifact_root)
    return artifact_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Agent Skills category map")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mvp.yaml",
        help="Pipeline YAML configuration",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU explicitly for development or smoke tests",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        run_path = run_pipeline(args.config.resolve(), use_cpu=args.cpu)
    except Exception as error:
        parser.exit(1, f"agsk: {error}\n")
    print(run_path)


if __name__ == "__main__":
    main()