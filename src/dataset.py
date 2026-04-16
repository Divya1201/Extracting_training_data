"""
Dataset utilities (§5)

CIFAR-10 experiments (§5):
  - 16 diffusion models, each trained on a RANDOMLY PARTITIONED 50% of CIFAR-10.
  - Paper §5: "we train 16 diffusion models, each on a randomly-partitioned
    half of the CIFAR-10 training dataset."
  - For MIA evaluation: each image appears as 'member' in ~8/16 models.

This module provides:
  - CIFAR-10 download and 50%-split generation.
  - Helper to get member / non-member sets for a given model index.
  - LAION deduplication scoring (§4.2) — requires a CLIP index.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10


# ---------------------------------------------------------------------------
# CIFAR-10 with 50% random splits (§5)
# ---------------------------------------------------------------------------

N_MODELS = 16  #2        # § "train 16 diffusion models"
SPLIT_FRACTION = 0.5     # § "randomly-partitioned half"


def prepare_cifar10(output_dir: Path, num_images: Optional[int] = None) -> Path:
    """
    Download CIFAR-10 and save images as PNG files.

    Args:
        output_dir: where to save.
        num_images: if None, save all 50,000 training images.

    Returns path to directory containing PNG images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR10(root=str(output_dir), train=True, download=True)

    save_dir = output_dir / "cifar10_images"
    save_dir.mkdir(exist_ok=True)

    n = min(num_images, len(dataset)) if num_images else len(dataset)
    for i in range(n):
        img, label = dataset[i]
        img.save(save_dir / f"img_{i:05d}_class{label}.png")

    print(f" Saved {n} CIFAR-10 images to {save_dir}")
    return save_dir


def generate_cifar10_splits(
    n_models: int = N_MODELS,
    total_images: int = 50_000,       #100
    split_fraction: float = SPLIT_FRACTION,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate n_models random 50/50 train/test splits of CIFAR-10 indices.

    Returns list of (member_indices, nonmember_indices) tuples.
    Each model is trained on member_indices only.

    Paper §5: "each on a randomly-partitioned half of the CIFAR-10 training
    dataset. We use half the dataset as is standard in privacy analyses."
    """
    rng = random.Random(seed)
    all_indices = list(range(total_images))
    split_size = int(total_images * split_fraction)

    splits = []
    for model_idx in range(n_models):
        shuffled = all_indices.copy()
        rng.shuffle(shuffled)
        members = set(shuffled[:split_size])
        nonmembers = set(shuffled[split_size:])
        splits.append((sorted(members), sorted(nonmembers)))
    return splits


def get_member_nonmember_images(
    image_dir: Path,
    splits: List[Tuple[List[int], List[int]]],
    model_idx: int,
) -> Tuple[List[Path], List[Path]]:
    """
    Return (member_paths, nonmember_paths) for a given model.

    Image filenames must be 'img_{i:05d}_class{label}.png'.
    """
    member_idxs, nonmember_idxs = splits[model_idx]
    all_images = sorted(image_dir.glob("img_*.png"))
    idx_to_path = {i: p for i, p in enumerate(all_images)}

    members = [idx_to_path[i] for i in member_idxs if i in idx_to_path]
    nonmembers = [idx_to_path[i] for i in nonmember_idxs if i in idx_to_path]
    return members, nonmembers



def get_top_duplicated_prompts(
    duplicate_counts: Dict[Path, int],
    captions: Dict[Path, str],
    top_k: int = 350_000,    #20
) -> List[str]:
    """
    Return the top_k captions corresponding to the most-duplicated images.

    Paper §4.2: "we select the 350,000 most-duplicated examples from the
    training dataset and generate 500 candidate images for each."
    """
    sorted_items = sorted(duplicate_counts.items(), key=lambda x: x[1], reverse=True)
    top_ids = [img_id for img_id, _ in sorted_items[:top_k]]
    return [captions[img_id] for img_id in top_ids if img_id in captions]
