"""
DP-LoRA integrated extraction attack runner.

This version FIXES:
✔ Proper clipping (no pre-normalization issue)
✔ LoRA-style transformation
✔ Integration with full extraction pipeline (attack.py)
✔ Applies DP in embedding space BEFORE attack

Pipeline:
generated images → CLIP embeddings → LoRA → DP → extraction attack
"""

from __future__ import annotations

import argparse
import pickle
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_utils import embed_directory
from attack import run_extraction_attack, AttackConfig

# ==============================
# Utility
# ==============================
def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


# ==============================
# LoRA Adapter
# ==============================
class LoRAAdapter:
    """
    LoRA-style transformation: v → v + A(Bv)
    (Simulates low-rank parameter adaptation)
    """

    def __init__(self, dim: int = 512, rank: int = 8, scale: float = 0.1):
        self.A = np.random.randn(dim, rank).astype(np.float32) * scale
        self.B = np.random.randn(rank, dim).astype(np.float32) * scale

    def forward(self, v: np.ndarray) -> np.ndarray:
        return v + self.A @ (self.B @ v)


# ==============================
# DP-LoRA Mechanism
# ==============================
def dp_lora_mechanism(
    embeddings: Dict[Path, np.ndarray],
    C: float = 0.5,          # IMPORTANT: <1 so clipping works
    sigma: float = 0.05,
    rank: int = 8,
    seed: int = 0, 
) -> Dict[Path, np.ndarray]:

    paths = list(embeddings.keys())
    vecs = np.stack([embeddings[p] for p in paths]).astype(np.float32)

    # --------------------------------
    # 1. Apply LoRA (NO normalization before clipping)
    # --------------------------------
    np.random.seed(seed) 
    adapter = LoRAAdapter(dim=vecs.shape[1], rank=rank)
    vecs_lora = np.array([adapter.forward(v) for v in vecs])

    # --------------------------------
    # 2. Clip 
    # --------------------------------
    norms = np.linalg.norm(vecs_lora, axis=1, keepdims=True)
    clip_factors = np.minimum(1.0, C / (norms + 1e-12))
    vecs_clipped = vecs_lora * clip_factors

    # --------------------------------
    # 3. Add DP noise
    # --------------------------------
    noise_std = sigma * C
    noise = np.random.normal(0, noise_std, size=vecs_clipped.shape).astype(np.float32)
    vecs_noisy = vecs_clipped + noise

    # --------------------------------
    # 4. Normalize (ONLY HERE)
    # --------------------------------
    vecs_out = _l2_normalize_rows(vecs_noisy)

    return {p: vecs_out[i] for i, p in enumerate(paths)}


# ==============================
# CLI
# ==============================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)

    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05])
    parser.add_argument("--rank", type=int, default=8)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--clique-min-size", type=int, default=10)
    parser.add_argument("--patch-l2-threshold", type=float, default=0.05)
    parser.add_argument("--extraction-delta", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # --------------------------------
    # Reference dataset
    # --------------------------------
    if args.reference_dir is None:
        raise ValueError("You must provide --reference-dir (LAION dataset)")
    else:
        reference_dir = args.reference_dir

    # --------------------------------
    # Embed GENERATED images 
    # --------------------------------
    cache_file = Path("clip_embeddings_combined.pkl")

    if cache_file.exists():
        print(" Loading cached CLIP embeddings...")
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print(" Embedding generated images with CLIP...")
        print("Embedding generated + reference images...")

        emb_gen = embed_directory(args.generated_dir)
        emb_ref = embed_directory(reference_dir)

        # Merge both
        embeddings = {**emb_gen, **emb_ref}
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)

    for sigma in args.sigmas:
        print(f"\n[DP-LoRA MODE | sigma={sigma}]")

        # --------------------------------
        # Apply DP-LoRA
        # --------------------------------
        dp_embeddings = dp_lora_mechanism(
            embeddings,
            C=args.c,
            sigma=sigma,
            rank=args.rank,
            seed=args.seed,
        )

        # --------------------------------
        # Run FULL extraction attack
        # --------------------------------
        config = AttackConfig(
            image_size=args.image_size,
            clique_min_size=args.clique_min_size,
            patch_l2_threshold=args.patch_l2_threshold,
            extraction_delta=args.extraction_delta,
            top_k=5,
            clip_cosine_threshold=None,  
        )

        results = run_extraction_attack(
            generated_dir=args.generated_dir,
            reference_dir=reference_dir,
            config=config,
            clip_embeddings=dp_embeddings,  
        )
        if not results:
            print("\n No results found (DP-LoRA)")
            print("Try:")
            print("  --clique-min-size 5")
            print("  --patch-l2-threshold 0.1")
            continue
            
        extracted = [r for r in results if r.extracted]

        # --------------------------------
        # Save results
        # --------------------------------
        output_path = Path(
            f"results_dp_lora_{sigma:.2f}.pkl" if len(args.sigmas) > 1 else "results_dp_lora.pkl"
        )

        with output_path.open("wb") as f:
            pickle.dump(results, f)

        # ----------------------------
        # Save CSV 
        # ----------------------------
        csv_path = output_path.with_suffix(".csv")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_path",
                "match_path",
                "l2_norm",
                "mean_clique_dist",
                "adaptive_score",
                "clique_size",
                "extracted",
            ])

            for r in results:
                writer.writerow([
                    str(r.query_path),
                    str(r.match_path),
                    f"{r.l2_norm:.8f}",
                    f"{r.mean_clique_dist:.8f}",
                    f"{r.adaptive_score:.8f}",
                    r.clique_size,
                    r.extracted,
                ])
                
        print(f" Extracted images: {len(extracted)}")
        print(f" Saved to {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
