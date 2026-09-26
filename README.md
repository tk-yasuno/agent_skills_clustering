# Agent Skills Category Map

A reproducible MVP that collects `name` and `description` from GitHub `SKILL.md` files, embeds them with E5-large-v2 on a CUDA GPU, clusters the vectors, and generates a self-contained Plotly map.

## Current status (v0.2.0, 2026-09-26)

The v0.2.0 end-to-end run completed on the configured 20 repositories. GitHub tree metadata reported 1,512 candidate `SKILL.md` files; 1,506 were valid and collected, and six were rejected or failed during fetch/frontmatter validation.

| Repository | Discovered | Collected | Failed | GitHub SPDX license |
| --- | ---: | ---: | ---: | --- |
| `omnitric/agent-skills` | 35 | 32 | 3 | Not reported |
| `bytedance/deer-flow` | 33 | 31 | 2 | MIT |
| `K-Dense-AI/scientific-agent-skills` | 166 | 166 | 0 | MIT |
| `lingzhi227/agent-research-skills` | 31 | 31 | 0 | Not reported |
| `agent-skills-hub/agent-skills-hub` | 812 | 811 | 1 | MIT |
| `anthropics/skills` | 20 | 20 | 0 | Not reported |
| `addyosmani/agent-skills` | 25 | 25 | 0 | MIT |
| `vercel-labs/agent-skills` | 9 | 9 | 0 | Not reported |
| `muratcankoylan/Agent-Skills-for-Context-Engineering` | 23 | 23 | 0 | MIT |
| `supabase/agent-skills` | 2 | 2 | 0 | MIT |
| `apify/agent-skills` | 5 | 5 | 0 | Not reported |
| `WordPress/agent-skills` | 19 | 19 | 0 | NOASSERTION |
| `openclaw/agent-skills` | 8 | 8 | 0 | MIT |
| `Kotlin/kotlin-agent-skills` | 10 | 10 | 0 | Apache-2.0 |
| `hashicorp/agent-skills` | 20 | 20 | 0 | MPL-2.0 |
| `MicrosoftDocs/Agent-Skills` | 202 | 202 | 0 | CC-BY-4.0 |
| `dbt-labs/dbt-agent-skills` | 16 | 16 | 0 | Apache-2.0 |
| `elastic/agent-skills` | 52 | 52 | 0 | Apache-2.0 |
| `firebase/agent-skills` | 13 | 13 | 0 | Apache-2.0 |
| `ClickHouse/agent-skills` | 11 | 11 | 0 | Apache-2.0 |

Run `20260926T111754504482Z` used `intfloat/e5-large-v2` revision `f169b11e22de13617baa190a028a32f3493550b6` on an NVIDIA GeForce RTX 4060 Ti with 16 GiB VRAM. The normalized embeddings are 1,024-dimensional. UMAP/HDBSCAN produced 82 clusters and 459 noise points (30.48%). The run's `skills.json` contains all 1,506 records with raw `SKILL.md` text; its size is approximately 18.3 MB. No raw field was missing.

Run artifacts are in `artifacts/20260926T111754504482Z/`; the standalone map is `artifacts/20260926T111754504482Z/index.html`. The run refreshed `dist/hf-space/`. v0.2 includes `GH_PAT`-first authentication with `GITHUB_TOKEN` fallback. The full test suite passes (14 tests). Hugging Face Space creation/upload and LinkedIn posting remain manual and have not been performed.

## Requirements

- Windows 10/11, Python 3.12, and a recent NVIDIA driver
- CUDA-capable NVIDIA GPU; the runtime records the detected model and VRAM
- A GitHub token is optional for a small public-repository run. For the configured 20 repositories, use a read-only token to avoid the unauthenticated core API limit.

## Environment setup (PowerShell)

```powershell
py -3.12 -m venv .venv-agsk
.\.venv-agsk\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the current Windows CUDA wheel selected by the official PyTorch selector. For CUDA 12.8, the command is:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

Check the GPU before a full run:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

If the selected CUDA wheel is incompatible with the installed driver, choose a compatible wheel using [the official PyTorch install selector](https://pytorch.org/get-started/locally/). Do not replace it with a CPU-only wheel for the production run.

## Run

Review `configs/sources.yaml` and `configs/mvp.yaml`, then run:

```powershell
$env:GH_PAT = "<GitHub token>"
agsk --verbose
```

`GH_PAT` is preferred; `GITHUB_TOKEN` is used only when `GH_PAT` is unset. The environment variable applies only to the current PowerShell session. Do not put the token in configuration files or commit it. The collector makes three GitHub API calls per repository plus one raw-content download for every discovered `SKILL.md`; request volume therefore depends on skill count, not only the number of repositories.

For local development or smoke tests only, CPU mode is explicit:

```powershell
agsk --cpu
```

The pipeline writes a timestamped run under `artifacts/`. Each run's `skills.json` is a JSON array with extracted metadata, provenance, content hash, and the original `SKILL.md` text. Embeddings and analysis stay in separate files. The pipeline also refreshes `dist/hf-space/`, which contains only a static Space `README.md`, `index.html`, and summary `manifest.json`; it does not include the raw skill records. HTML includes Plotly JavaScript and can be opened offline.

## Publish manually

1. Create a **Static** Space on Hugging Face.
2. Upload the contents of `dist/hf-space/` to the Space repository root.
3. Open the Space URL and verify the map and source links.
4. Review `artifacts/<run-id>/linkedin_draft.md`; replace the map URL and confirm all counts before posting.

Descriptions are included in hover text only when the repository reports an SPDX license on the reuse allowlist. This is a conservative display check, not legal advice; verify file-level licensing before redistribution. Source repository and commit links are retained in the map and run manifest.

HDBSCAN cluster IDs are exploratory labels, not semantic category names. The map reflects only the configured repositories and is not a census of all Agent Skills. For repeatability, retain each run's manifest, source commit SHAs, seed, model name, and package versions.

## License

This project's source code is licensed under the MIT License; see [LICENSE](LICENSE). That license does not change the licenses or reuse terms of the upstream repositories or collected skill content.