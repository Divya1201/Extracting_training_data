"""
CLIP embedding utilities (§4.2)

Paper: "we first embed each image to a 512-dimensional vector using CLIP,
and then perform the all-pairs comparison between images in this lower-
dimensional space (increasing efficiency by over 1500×)."

Model: ViT-B/32 pretrained on OpenAI data (same as paper).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paper §4.2: CLIP ViT-B/32 (512-dimensional embeddings)
_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
_model = _model.to(device).eval()


def get_clip_embedding(image_path: Path) -> np.ndarray:
    """
    Embed a single image with CLIP ViT-B/32.
    Returns a unit-normalised 512-dim float32 vector.
    """
    image = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype(np.float32)


def embed_directory(
    directory: Path,
    image_size: int = 512,
) -> Dict[Path, np.ndarray]:
    """
    Embed all images in a directory.
    Returns dict mapping Path → 512-dim embedding.
    """
    paths: List[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        paths.extend(directory.rglob(suffix))
    paths = sorted(paths)

    embeddings: Dict[Path, np.ndarray] = {}

    for i, p in enumerate(paths):
        try:
            embeddings[p] = get_clip_embedding(p)
        except Exception as e:
            print(f" Skipping {p}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Embedded {i+1}/{len(paths)}")

    return embeddings


def find_near_duplicates(
    embeddings: Dict[Path, np.ndarray],
    cosine_threshold: float = 0.9,
    batch_size: int = 1000,   #200
) -> Dict[Path, List[Path]]:
    """
    Efficient block-wise duplicate detection using cosine similarity.
    Avoids full NxN matrix computation.
    """

    print(" Using block-wise cosine similarity search...")

    paths = sorted(embeddings.keys())
    vecs = np.stack([embeddings[p] for p in paths])  # already normalized

    n = len(vecs)
    duplicates: Dict[Path, List[Path]] = {p: [] for p in paths}

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        print(f"Processing batch {i} → {end} / {n}")

        batch = vecs[i:end]  # (batch_size × 512)

        # Compute similarity (batch vs all)
        sims = batch @ vecs.T   # (batch_size × n)

        for bi in range(batch.shape[0]):
            idx_i = i + bi

            # Get all indices above threshold
            similar_indices = np.where(sims[bi] >= cosine_threshold)[0]

            for idx_j in similar_indices:
                # Avoid duplicate/self comparisons
                if idx_j > idx_i:
                    p1 = paths[idx_i]
                    p2 = paths[idx_j]

                    duplicates[p1].append(p2)
                    duplicates[p2].append(p1)

    return duplicates
