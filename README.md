# Agent Skills Category Map

A reproducible MVP that collects `name` and `description` from GitHub `SKILL.md` files, embeds them with E5-large-v2 on a CUDA GPU, clusters the vectors, and generates a self-contained Plotly map.

## Current status (v0.2.2, 2026-09-26)

The v0.2.2 end-to-end run completed against all 250 configured repositories and exceeded the 10,000-valid-skill target. GitHub trees contained 12,634 candidate `SKILL.md` files; 11,738 valid records were collected. All 250 repository API/tree scans succeeded. Eight hundred ninety-six individual files were excluded because required frontmatter was missing or invalid.

| Collection result | Count |
| --- | ---: |
| Repositories configured and scanned | 250 |
| `SKILL.md` files discovered | 12,634 |
| Skill records collected | 11,738 |
| Individual skill files rejected | 896 |
| Repository-level scan errors | 0 |
| Missing `raw` fields in `skills.json` | 0 |
| Collection target | 10,000 valid skills |

The 896 rejected files comprised 667 missing `name` fields, 226 invalid YAML frontmatter blocks, and 3 missing `description` fields. Rejections were concentrated in `alirezarezvani/claude-skills` (458), `euwebertdefreitas/ai-skills-for-claude-code` (180), `nexscope-ai/eCommerce-Skills` (42), `EliasOulkadi/shokunin` (39), and `nexscope-ai/nexscope-ecommerce-skills` (35). No repository-level API/tree errors occurred.

Run `20260926T135448595005Z` used `intfloat/e5-large-v2` revision `f169b11e22de13617baa190a028a32f3493550b6` on an NVIDIA GeForce RTX 4060 Ti with 16 GiB VRAM. The normalized embeddings are 1,024-dimensional. UMAP/HDBSCAN produced 512 clusters and 2,946 noise points (25.10%). The run's `skills.json` contains all 11,738 records with original `SKILL.md` text and is approximately 126.2 MB.

UMAP's spectral initialization warned about a small eigengap and fell back to random initialization; the full run completed and generated the map. Treat the 512 HDBSCAN labels as exploratory groups, not validated semantic categories.

Run artifacts are in `artifacts/20260926T135448595005Z/`; the standalone map is `artifacts/20260926T135448595005Z/index.html`. The run refreshed `dist/hf-space/`. The full test suite passes (14 tests). Hugging Face Space creation/upload and LinkedIn posting remain manual and have not been performed.

## Requirements

- Windows 10/11, Python 3.12, and a recent NVIDIA driver
- CUDA-capable NVIDIA GPU; the runtime records the detected model and VRAM
- A GitHub token is required for the configured 250 repositories to avoid the unauthenticated core API limit.

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