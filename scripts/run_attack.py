"""
run_attack.py — CLI entry-point for the generate-and-filter extraction pipeline.

Implements §4.2.1 of Carlini et al. 2023:
  1. Scan a directory of generated images for memorised cliques.
  2. Match each clique representative against the reference dataset.
  3. Rank results by adaptive L2 score (§5.1) and save to CSV.
  4. Optionally display top (generated, match) pairs (Figure 3 style).

Usage examples
--------------
# Basic run against your own generated/reference directories
python run_attack.py \
    --generated-dir outputs/generated \
    --reference-dir data

# Use CIFAR-10 as the reference set (auto-downloads)
python run_attack.py \
    --generated-dir outputs/generated \

# Tune thresholds to match the paper's Stable Diffusion settings
python run_attack.py \
    --generated-dir outputs/sd_generated \
    --reference-dir data/laion_sample \
    --image-size 512 \
    --clique-min-size 10 \
    --patch-l2-threshold 0.05 \
    --extraction-delta 0.15
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attack import AttackConfig, AttackResult, run_extraction_attack
from clip_utils import embed_directory
from visualize import show_pair, show_top_results


# =============================================================================
# CLI argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diffusion-model training-data extraction attack "
            "(Carlini et al. 2023, §4.2.1 generate-and-filter pipeline)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / Output ────────────────────────────────────────────────────────
    io = parser.add_argument_group("I/O paths")
    io.add_argument(
        "--generated-dir", type=Path, required=True,
    )
    io.add_argument(
        "--reference-dir", type=Path, required=True,
        help="Directory containing training-set reference images.",
    )
    io.add_argument(
        "--output", type=Path, default=Path("attack_results.csv"),
        help="Path to write the CSV results file.",
    )


    # ── AttackConfig knobs ────────────────────────────────────────────────────
    cfg = parser.add_argument_group("Attack configuration (AttackConfig)")
    cfg.add_argument(
        "--image-size", type=int, default=512,
        help=(
            "Resize all images to this square resolution before comparison. "
            "Paper: 512×512 for Stable Diffusion (§4.2), 32×32 native for CIFAR-10 (§5)."
        ),
    )
    cfg.add_argument(
        "--top-k", type=int, default=5,
        help="Number of nearest reference matches to record per clique image.",
    )
    cfg.add_argument(
        "--no-normalize", action="store_true",
        help="Disable normalisation of pixel values to [0, 1].",
    )

    # Tile / patch-L2 (§4.2.1)
    cfg.add_argument(
        "--tile-size", type=int, default=128,
        help=(
            "Tile side length for patch-L2 (§4.2.1). "
            "Paper: 128 px → 16 tiles on a 512×512 image."
        ),
    )
    cfg.add_argument(
        "--patch-l2-threshold", type=float, default=0.05,
        help=(
            "Two generated images are connected by an edge if their "
            "patch-L2 < this value (§4.2.1)."
        ),
    )

    # Clique detection (§4.2.1)
    cfg.add_argument(
        "--clique-min-size", type=int, default=10,
        help=(
            "Minimum clique/component size to flag as memorised (§4.2.1). "
            'Paper: "at least size 10" out of 500 generations per prompt.'
        ),
    )

    # CLIP pre-filter (§4.2)
    cfg.add_argument(
        "--clip-cosine-threshold", type=float, default=None,
        help=(
            "CLIP cosine-similarity threshold for near-duplicate pre-filtering (§4.2). "
            'Paper: "high cosine similarity".'
        ),
    )
    cfg.add_argument(
        "--clip-cache", type=Path, default=Path("clip_embeddings_generated.pkl"),
        help="Cache path for generated-image CLIP embeddings.",
    )

    # Adaptive L2 score (§5.1)
    cfg.add_argument(
        "--adaptive-alpha", type=float, default=0.5,
        help="α in the adaptive L2 score formula (§5.1). Paper: α = 0.5.",
    )
    cfg.add_argument(
        "--adaptive-n", type=int, default=50,
        help="n nearest neighbours used in the adaptive L2 score (§5.1). Paper: n = 50.",
    )

    # Extraction criterion (Definition 1, §4.1)
    cfg.add_argument(
        "--extraction-delta", type=float, default=0.15,
        help=(
            "δ threshold for (ℓ₂, δ)-Diffusion Extraction (Definition 1, §4.1). "
            "Paper reports 94 images extracted at δ = 0.15 from Stable Diffusion."
        ),
    )
    cfg.add_argument(
        "--extraction-rule",
        choices=("delta", "adaptive_delta"),
        default="delta",
        help=(
            "Use 'delta' for the Stable Diffusion paper extraction rule. "
            "Use 'adaptive_delta' for the CIFAR §5.1 adaptive-L2 variant."
        ),
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    vis = parser.add_argument_group("Visualisation")
    vis.add_argument(
        "--show-top", type=int, default=3,
        help="Number of top extracted pairs to display after the run (0 = skip).",
    )
    vis.add_argument(
        "--show-grid", action="store_true",
        help="Display a grid of all extracted results (show_top_results).",
    )

    args = parser.parse_args()

    return args


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    print("\n Running Extraction Attack...")
    print(f"Generated dir : {args.generated_dir}")
    print(f"Reference dir : {args.reference_dir}")

    # ── Build AttackConfig from CLI args ──────────────────────────────────────
    config = AttackConfig(
        image_size=args.image_size,
        top_k=args.top_k,
        normalize=not args.no_normalize,
        tile_size=args.tile_size,
        patch_l2_threshold=args.patch_l2_threshold,
        clique_min_size=args.clique_min_size,
        clip_cosine_threshold=(
            args.clip_cosine_threshold
            if args.clip_cosine_threshold is not None
            else AttackConfig.clip_cosine_threshold
        ),
        adaptive_alpha=args.adaptive_alpha,
        adaptive_n=args.adaptive_n,
        extraction_delta=args.extraction_delta,
        extraction_rule=args.extraction_rule,
    )

    print(" Running memorisation extraction attack (Carlini et al. 2023 §4.2.1)")
    print(f"   Generated dir : {args.generated_dir}")
    print(f"   Reference dir : {args.reference_dir}")
    print(f"   Image size    : {config.image_size}×{config.image_size}")
    print(f"   Clique min    : {config.clique_min_size}")
    print(f"   Patch-L2 τ    : {config.patch_l2_threshold}")
    print(f"   Extraction δ  : {config.extraction_delta}")
    print(f"   Rule          : {config.extraction_rule}")

    clip_embeddings = None
    if args.clip_cosine_threshold is not None:
        if args.clip_cache.exists():
            print(f" Loading CLIP embeddings from {args.clip_cache}")
            with args.clip_cache.open("rb") as f:
                clip_embeddings = pickle.load(f)
        else:
            print(" Embedding generated images with CLIP for the paper pre-filter...")
            clip_embeddings = embed_directory(args.generated_dir)
            with args.clip_cache.open("wb") as f:
                pickle.dump(clip_embeddings, f)

    # ── Run the attack ──────────────────────────────────────────────────────
    results: list[AttackResult] = run_extraction_attack(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        config=config,
        clip_embeddings=clip_embeddings,
    )

    if not results:
      print("\n❌ No suspicious samples found")

      print("\nTry these settings:")
      print("  --clique-min-size 5")
      print("  --patch-l2-threshold 0.1")

      print("\nAlso check:")
      print("  ✔ generated images exist")
      print("  ✔ reference dataset is not empty")

      return

    # ── Console summary ───────────────────────────────────────────────────────
    extracted = [r for r in results if r.extracted]
    print(f"\n{'='*60}")
    print(f"  Total candidate pairs  : {len(results)}")
    print(f"  Extracted (Def. 1, δ={config.extraction_delta}) : {len(extracted)}")
    print(f"{'='*60}")

    print("\n TOP SUSPICIOUS SAMPLES (sorted by adaptive L2 score ↑):")
    for r in results[:5]:
        print(
            f"  {r.query_path.name} → {r.match_path.name}"
            f" | L2={r.l2_norm:.4f}"
            f" | adaptive_score={r.adaptive_score:.4f}"
            f" | clique_size={r.clique_size}"
            f" | extracted={r.extracted}"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header mirrors AttackResult fields (§4.1 Definition 1 + §5.1)
        writer.writerow([
            "query_path",       # path to the generated image
            "match_path",       # path to the nearest reference image
            "l2_norm",          # normalised L2 (Definition 1, §4.1)
            "mean_clique_dist", # mean intra-clique patch-L2 (§4.2.1 ranking)
            "adaptive_score",   # §5.1 score; < 1.0 → extraction declared
            "clique_size",      # number of near-identical generations (§4.2.1)
            "extracted",        # True if (ℓ₂,δ)-extracted (Definition 1)
        ])
        for result in results:
            writer.writerow([
                str(result.query_path),
                str(result.match_path),
                f"{result.l2_norm:.8f}",
                f"{result.mean_clique_dist:.8f}",
                f"{result.adaptive_score:.8f}",
                result.clique_size,
                result.extracted,
            ])

    print(f"\n Saved {len(results)} rows → {args.output}")


    with open("results_normal.pkl", "wb") as f:
        pickle.dump(results, f)

    print(" Saved results_normal.pkl")

    # ── Visualisation (Figure 3 style) ────────────────────────────────────────
    if args.show_top > 0:
        print(f"\n  Displaying top {args.show_top} (generated, match) pairs...")
        for r in results[: args.show_top]:
            title = (
                f"L2={r.l2_norm:.4f}  score={r.adaptive_score:.4f}"
                f"  clique={r.clique_size}  extracted={r.extracted}"
            )
            show_pair(r.query_path, r.match_path, title=title)

    if args.show_grid:
        print("\n  Displaying extracted-image grid...")
        show_top_results(results, n=len(results), only_extracted=True)


if __name__ == "__main__":
    main()
