from typing import Any
from importlib.metadata import version

import numpy as np


def cuda_runtime() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Install a CUDA-enabled PyTorch wheel and "
            "verify the NVIDIA driver, or use --cpu for a development run."
        )
    device = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(device.total_memory / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def embed_texts(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    seed: int,
    use_cpu: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    if not texts:
        raise ValueError("No skill descriptions were provided for embedding")

    if use_cpu:
        device = "cpu"
        runtime = {"device": device, "gpu_name": None, "gpu_memory_gib": None}
    else:
        runtime = cuda_runtime()
        device = "cuda"

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SentenceTransformer(model_name, device=device, trust_remote_code=False)
    embeddings = model.encode(
        [f"passage: {text}" for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(texts), 1024):
        raise RuntimeError(
            f"Expected {len(texts)} normalized 1024D embeddings; got {embeddings.shape}"
        )
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
        raise RuntimeError("Embedding model returned non-normalized vectors")
    transformer = getattr(model[0], "auto_model", None)
    model_config = getattr(transformer, "config", None)
    runtime.update(
        {
            "model_name": model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "resolved_model_revision": getattr(model_config, "_commit_hash", None),
            "sentence_transformers_version": version("sentence-transformers"),
        }
    )
    return embeddings, runtime