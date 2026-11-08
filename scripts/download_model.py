"""One-time download of the embedding model into the local Hugging Face cache (needs network).

After this succeeds, set HF_HUB_OFFLINE=1 and everything runs offline.

Run:  python -m scripts.download_model
"""

from __future__ import annotations

import app.config as config


def main() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"Cached {config.EMBEDDING_MODEL} (dimension {dim}). You can now set HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
