"""
Visualisation utilities (§4.2.2 — Figure 3 style).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from PIL import Image

from attack import AttackResult


def show_pair(
    gen_path: Path,
    ref_path: Path,
    title: str = "",
) -> None:
    """
    Show a generated image next to its nearest training-set match.
    Reproduces the visual from Figure 3 in the paper.
    """
    gen = Image.open(gen_path)
    ref = Image.open(ref_path)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(gen);  axes[0].set_title("Generated");  axes[0].axis("off")
    axes[1].imshow(ref);  axes[1].set_title("Training set match"); axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def show_top_results(
    results: List[AttackResult],
    n: int = 9,
    only_extracted: bool = True,
) -> None:
    """
    Show a grid of (generated, match) pairs, sorted by adaptive_score.
    """
    filtered = [r for r in results if r.extracted] if only_extracted else results
    filtered = filtered[:n]

    if not filtered:
        print("No extracted images to display.")
        return

    cols = 4   # pairs per row (2 images per pair)
    rows = (len(filtered) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
    axes = axes.reshape(-1, 2)

    for i, result in enumerate(filtered):
        gen = Image.open(result.query_path)
        ref = Image.open(result.match_path)
        axes[i, 0].imshow(gen);  axes[i, 0].axis("off")
        axes[i, 1].imshow(ref);  axes[i, 1].axis("off")
        axes[i, 0].set_title(f"Gen  L2={result.l2_norm:.3f}", fontsize=7)
        axes[i, 1].set_title(f"Match score={result.adaptive_score:.3f}", fontsize=7)

    for j in range(len(filtered), len(axes)):
        axes[j, 0].axis("off")
        axes[j, 1].axis("off")

    plt.suptitle("Extracted training images (§4.2.2)", fontsize=11)
    plt.tight_layout()
    plt.show()
