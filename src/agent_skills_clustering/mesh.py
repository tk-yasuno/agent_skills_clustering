from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from agent_skills_clustering.licensing import REUSABLE_LICENSES
from agent_skills_clustering.models import SkillRecord

LABEL_STOP_WORDS = ENGLISH_STOP_WORDS | {
    "agent",
    "agents",
    "claude",
    "code",
    "description",
    "descriptions",
    "help",
    "helps",
    "including",
    "instructions",
    "llm",
    "phrases",
    "plugin",
    "plugins",
    "provides",
    "source",
    "sources",
    "skill",
    "skills",
    "tool",
    "tools",
    "use",
    "used",
    "user",
    "users",
    "using",
    "workflow",
    "workflows",
    "says",
    "implements",
    "include",
    "includes",
    "fails",
    "example",
    "examples",
}
LABEL_ACRONYMS = {
    "api",
    "aws",
    "cli",
    "css",
    "gpu",
    "html",
    "mcp",
    "pdf",
    "pptx",
    "sql",
    "ui",
    "ux",
}
TOPIC_RULES = (
    ("Security & Safety", {"security", "vulnerability", "jailbreak", "destructive", "risk", "guardrails"}),
    ("Video & Audio Production", {"video", "audio", "voice", "hyperframes", "music", "media"}),
    ("E-commerce & Growth", {"shop", "shopify", "tiktok", "ecommerce", "marketing", "seo", "conversion"}),
    ("Finance & Market Analysis", {"finance", "financial", "market", "berkshire", "investment", "stock"}),
    ("Research & Literature", {"research", "literature", "academic", "paper", "patent", "citation"}),
    ("Office Documents & Slides", {"pdf", "pptx", "docx", "xlsx", "office", "deck", "spreadsheet", "formulas"}),
    ("Memory & Meeting Notes", {"memory", "memories", "remember", "meeting", "handoff", "recall"}),
    ("Data Center & Model Infrastructure", {"nemo", "megatron", "nvidia", "doca", "bluefield", "tao", "cuopt", "inference", "training"}),
    ("Frontend & Interface Design", {"react", "frontend", "interface", "css", "ui", "ux", "design"}),
    ("GitHub & Code Review", {"github", "pull", "commit", "review", "actions", "merge"}),
    ("Databases & Analytics", {"clickhouse", "postgres", "database", "sql", "qdrant", "analytics"}),
    ("Cloud-Native Operations", {"helm", "kubernetes", "docker", "deploy", "devops", "terraform"}),
    ("Command-Line Tools", {"command", "cli", "terminal", "shell", "invoke"}),
    ("Knowledge & Vector Search", {"wiki", "obsidian", "vault", "vector", "retrieval", "search"}),
    ("Software Architecture & Planning", {"specification", "architecture", "implementation", "requirements", "planning"}),
    ("Education & Tutoring", {"socratic", "teach", "learning", "tutor", "explain"}),
    ("Autonomous Workflows", {"loop", "autonomous", "automation", "workflow", "orchestration"}),
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _normalized_entropy(counts: list[int], category_count: int) -> float:
    nonzero = [count for count in counts if count > 0]
    if len(nonzero) < 2 or category_count < 2:
        return 0.0
    probabilities = np.asarray(nonzero, dtype=np.float64) / sum(nonzero)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(category_count))


def _display_keyword(keyword: str) -> str:
    return " ".join(
        word.upper() if word in LABEL_ACRONYMS else word.capitalize()
        for word in keyword.replace("-", " ").split()
    )


def _human_readable_label(keywords: list[str], fallback_slug: str) -> tuple[str, str]:
    normalized_words = {
        word.lower()
        for keyword in keywords
        for word in keyword.replace("-", " ").split()
    }
    for category, clues in TOPIC_RULES:
        matched = [keyword for keyword in keywords if normalized_words.intersection(clues) and any(
            word.lower() in clues
            for word in keyword.replace("-", " ").split()
        )]
        if matched:
            specific = matched[0]
            return f"{category} · {_display_keyword(specific)}", category
    if keywords:
        readable_keywords = " & ".join(
            _display_keyword(keyword) for keyword in keywords[:2]
        )
        return readable_keywords, "license-filtered TF-IDF keywords"

    slug_words = fallback_slug.replace("_", "-").split("-")
    if slug_words and slug_words[-1].lower() == "skills":
        slug_words.pop()
    fallback = _display_keyword(" ".join(slug_words)) if slug_words else "Repository"
    return f"{fallback} skills", "Repository-name fallback"


def _canonical_keyword(keyword: str) -> str:
    word = keyword.lower()
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]
    return word


def _overlaps_existing_keyword(keyword: str, selected: list[str]) -> bool:
    candidate_words = {_canonical_keyword(word) for word in keyword.split()}
    for existing in selected:
        existing_words = {_canonical_keyword(word) for word in existing.split()}
        if candidate_words <= existing_words or existing_words <= candidate_words:
            return True
        for candidate_word in candidate_words:
            for existing_word in existing_words:
                shorter, longer = sorted(
                    (candidate_word, existing_word), key=len
                )
                if len(shorter) >= 6 and longer.startswith(shorter[:6]):
                    return True
    return False


def _cluster_topics(
    records: list[SkillRecord], macro_labels: np.ndarray, macro_count: int
) -> dict[int, list[str]]:
    eligible_indexes = [
        index
        for index, record in enumerate(records)
        if record.license_spdx in REUSABLE_LICENSES and macro_labels[index] >= 0
    ]
    if len(eligible_indexes) < 3:
        return {}

    documents = [
        f"{records[index].name}. {records[index].description}"
        for index in eligible_indexes
    ]
    document_labels = macro_labels[eligible_indexes]
    vectorizer = TfidfVectorizer(
        stop_words=list(LABEL_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=30_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#._-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {}
    terms = vectorizer.get_feature_names_out()
    corpus_mean = np.asarray(matrix.mean(axis=0)).ravel()
    topics: dict[int, list[str]] = {}

    for macro_label in range(macro_count):
        selected = document_labels == macro_label
        if not selected.any() or selected.all():
            continue
        in_cluster = np.asarray(matrix[selected].mean(axis=0)).ravel()
        outside_cluster = np.asarray(matrix[~selected].mean(axis=0)).ravel()
        scores = in_cluster - outside_cluster
        ranked_indexes = np.argsort(scores)[::-1]
        selected_terms: list[str] = []
        for term_index in ranked_indexes:
            if scores[term_index] <= 0:
                break
            term = str(terms[term_index])
            tokens = set(term.split())
            if any(word.isdigit() or any(char.isdigit() for char in word) for word in tokens):
                continue
            if _overlaps_existing_keyword(term, selected_terms):
                continue
            selected_terms.append(term)
            if len(selected_terms) == 3:
                break
        if selected_terms:
            topics[macro_label] = selected_terms
    return topics


def build_skill_mesh(
    records: list[SkillRecord],
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    target_cluster_count: int = 100,
    repository_cluster_links: int = 8,
    cluster_neighbors: int = 3,
) -> dict[str, Any]:
    skill_count = len(records)
    if embeddings.ndim != 2 or len(embeddings) != skill_count:
        raise ValueError("Embedding rows must match the number of skill records")
    if coordinates.shape != (skill_count, 2):
        raise ValueError("UMAP coordinates must have shape (skills, 2)")
    if labels.shape != (skill_count,):
        raise ValueError("HDBSCAN labels must have one value per skill")
    if target_cluster_count < 1:
        raise ValueError("target_cluster_count must be at least 1")

    original_labels = sorted(set(int(label) for label in labels if label >= 0))
    original_to_macro: dict[int, int] = {}
    macro_count = min(target_cluster_count, len(original_labels))
    if original_labels:
        original_centroids = np.vstack(
            [embeddings[labels == label].mean(axis=0) for label in original_labels]
        )
        original_centroids = np.vstack([_normalize(row) for row in original_centroids])
        if len(original_labels) <= target_cluster_count:
            macro_labels = np.arange(len(original_labels), dtype=np.int32)
        else:
            macro_labels = AgglomerativeClustering(
                n_clusters=target_cluster_count,
                metric="cosine",
                linkage="average",
            ).fit_predict(original_centroids)
        original_to_macro = {
            original_label: int(macro_label)
            for original_label, macro_label in zip(original_labels, macro_labels)
        }

    skill_macro_labels = np.asarray(
        [original_to_macro.get(int(label), -1) for label in labels], dtype=np.int32
    )
    repository_names = sorted({record.repo for record in records})
    repository_indexes = {
        repo: np.asarray(
            [index for index, record in enumerate(records) if record.repo == repo],
            dtype=np.int32,
        )
        for repo in repository_names
    }
    macro_indexes = {
        macro_label: np.flatnonzero(skill_macro_labels == macro_label)
        for macro_label in range(macro_count)
    }
    macro_topics = _cluster_topics(records, skill_macro_labels, macro_count)

    nodes: list[dict[str, Any]] = []
    macro_centroids: list[np.ndarray] = []
    macro_ids: list[int] = []
    for macro_label, indexes in macro_indexes.items():
        if not len(indexes):
            continue
        repository_counts = Counter(records[index].repo for index in indexes)
        source_entropy = _normalized_entropy(
            list(repository_counts.values()), len(repository_names)
        )
        center = _normalize(embeddings[indexes].mean(axis=0))
        macro_centroids.append(center)
        macro_ids.append(macro_label)
        topic_keywords = macro_topics.get(macro_label, [])
        fallback_repo = repository_counts.most_common(1)[0][0].split("/", 1)[-1]
        human_label, label_source = _human_readable_label(topic_keywords, fallback_repo)
        nodes.append(
            {
                "id": f"cluster:{macro_label}",
                "type": "macro_cluster",
                "label": human_label,
                "label_keywords": topic_keywords,
                "label_method": label_source,
                "x": float(coordinates[indexes, 0].mean()),
                "y": float(coordinates[indexes, 1].mean()),
                "skill_count": int(len(indexes)),
                "source_count": len(repository_counts),
                "source_diversity": source_entropy,
                "hdbscan_cluster_count": len(
                    set(int(label) for label in labels[indexes] if label >= 0)
                ),
                "top_repositories": [
                    {"repo": repo, "skill_count": count}
                    for repo, count in repository_counts.most_common(5)
                ],
            }
        )

    noise_indexes = np.flatnonzero(skill_macro_labels == -1)
    if len(noise_indexes):
        noise_repositories = Counter(records[index].repo for index in noise_indexes)
        nodes.append(
            {
                "id": "cluster:noise",
                "type": "noise",
                "label": "Unclustered (HDBSCAN noise)",
                "x": float(coordinates[noise_indexes, 0].mean()),
                "y": float(coordinates[noise_indexes, 1].mean()),
                "skill_count": int(len(noise_indexes)),
                "source_count": len(noise_repositories),
                "source_diversity": _normalized_entropy(
                    list(noise_repositories.values()), len(repository_names)
                ),
                "top_repositories": [
                    {"repo": repo, "skill_count": count}
                    for repo, count in noise_repositories.most_common(5)
                ],
            }
        )

    edges: list[dict[str, Any]] = []
    for repo, indexes in repository_indexes.items():
        macro_counts = Counter(int(skill_macro_labels[index]) for index in indexes)
        non_noise_counts = [count for label, count in macro_counts.items() if label >= 0]
        richness = _normalized_entropy(non_noise_counts, macro_count)
        represented = sorted(label for label in macro_counts if label >= 0)
        noise_count = macro_counts.get(-1, 0)
        top_clusters = [
            {"macro_cluster": label, "skill_count": count}
            for label, count in macro_counts.most_common()
            if label >= 0
        ][:repository_cluster_links]
        nodes.append(
            {
                "id": f"repo:{repo}",
                "type": "repository",
                "label": repo,
                "source_url": f"https://github.com/{repo}",
                "x": float(coordinates[indexes, 0].mean()),
                "y": float(coordinates[indexes, 1].mean()),
                "skill_count": int(len(indexes)),
                "macro_cluster_count": len(represented),
                "macro_cluster_coverage": (
                    len(represented) / macro_count if macro_count else 0.0
                ),
                "richness": richness,
                "noise_count": noise_count,
                "top_macro_clusters": top_clusters,
            }
        )
        for label, count in macro_counts.most_common():
            if label >= 0 and label not in {
                item["macro_cluster"] for item in top_clusters
            }:
                continue
            target = "cluster:noise" if label == -1 else f"cluster:{label}"
            edges.append(
                {
                    "source": f"repo:{repo}",
                    "target": target,
                    "type": "membership",
                    "weight": int(count),
                    "share": float(count / len(indexes)),
                }
            )

    if len(macro_centroids) > 1:
        centers = np.vstack(macro_centroids)
        similarity = centers @ centers.T
        neighbor_pairs: set[tuple[int, int]] = set()
        neighbor_count = min(cluster_neighbors, len(macro_ids) - 1)
        for row, macro_label in enumerate(macro_ids):
            nearest = np.argsort(similarity[row])[::-1]
            chosen = [index for index in nearest if index != row][:neighbor_count]
            for column in chosen:
                left, right = sorted((macro_label, macro_ids[column]))
                neighbor_pairs.add((left, right))
        for left, right in sorted(neighbor_pairs):
            left_row = macro_ids.index(left)
            right_row = macro_ids.index(right)
            edges.append(
                {
                    "source": f"cluster:{left}",
                    "target": f"cluster:{right}",
                    "type": "similarity",
                    "weight": float(similarity[left_row, right_row]),
                }
            )

    repository_richness = [node["richness"] for node in nodes if node["type"] == "repository"]
    noise_count = int(len(noise_indexes))
    summary = {
        "method": "average-linkage cosine agglomeration of HDBSCAN cluster centroids",
        "label_method": "rule-assisted license-filtered TF-IDF topics; exploratory",
        "labeled_macro_cluster_count": len(macro_topics),
        "repository_fallback_label_count": sum(
            node["type"] == "macro_cluster"
            and node["label_method"] == "Repository-name fallback"
            for node in nodes
        ),
        "target_macro_cluster_count": target_cluster_count,
        "macro_cluster_count": len(macro_ids),
        "source_hdbscan_cluster_count": len(original_labels),
        "noise_count": noise_count,
        "repository_count": len(repository_names),
        "skill_count": skill_count,
        "membership_edge_count": sum(edge["type"] == "membership" for edge in edges),
        "similarity_edge_count": sum(edge["type"] == "similarity" for edge in edges),
        "mean_repository_richness": (
            float(np.mean(repository_richness)) if repository_richness else 0.0
        ),
    }
    return {
        "summary": summary,
        "nodes": nodes,
        "edges": edges,
    }