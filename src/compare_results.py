"""Compare memorization-attack results for normal vs DP-LoRA models."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Tuple[float, Dict[str, Any], Path]]]:
    """Load baseline and DP-LoRA result files from ``results_dir``.

    Returns:
        (normal_results, dp_lora_results, dp_sigma_results)
    """

    def _load_pickle(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            print(f"[WARNING] Missing file: {path}")
            return None
        try:
            with path.open("rb") as file:
                return pickle.load(file)
        except Exception as exc:  # pragma: no cover - defensive for malformed files
            print(f"[WARNING] Failed to load {path}: {exc}")
            return None

    normal_path = results_dir / "results_normal.pkl"
    dp_lora_path = results_dir / "results_dp_lora.pkl"

    normal_results = _load_pickle(normal_path)
    dp_lora_results = _load_pickle(dp_lora_path)

    dp_sigma_results: List[Tuple[float, Dict[str, Any], Path]] = []
    for candidate in sorted(results_dir.glob("results_dp_lora_*.pkl")):
        result = _load_pickle(candidate)
        if result is None:
            continue

        sigma_match = re.search(r"results_dp_lora_([0-9]*\.?[0-9]+)", candidate.stem)
        if sigma_match is None:
            print(f"[WARNING] Could not parse sigma from filename: {candidate.name}")
            continue

        sigma = float(sigma_match.group(1))
        dp_sigma_results.append((sigma, result, candidate))

    return normal_results, dp_lora_results, dp_sigma_results


def compute_metrics(results: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Extract key metrics from one results dictionary."""
    if results is None:
        return None

    if isinstance(results, list):
        if len(results) == 0:
            return None

        extracted = [r for r in results if r.extracted]

        return {
            "avg_duplicates": len(extracted),
            "total_embeddings": len(results),
        }

    # fallback (old format)
    try:
        return {
            "avg_duplicates": float(results["avg_duplicates"]),
            "total_embeddings": int(results["total_embeddings"]),
        }
    except Exception as e:
        print(f"[WARNING] Could not extract metrics: {e}")
        return None


def plot_results(
    normal_metrics: Optional[Dict[str, float]],
    dp_metrics: Optional[Dict[str, float]],
    sigma_metrics: List[Tuple[float, Dict[str, float]]],
    output_dir: Path,
) -> None:
    """Create comparison plots for average duplicates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if normal_metrics is not None and dp_metrics is not None:
        labels = ["Normal", "DP-LoRA"]
        values = np.array(
            [
                normal_metrics["avg_duplicates"],
                dp_metrics["avg_duplicates"],
            ],
            dtype=float,
        )

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=["tab:blue", "tab:orange"])
        plt.ylabel("Average duplicates")
        plt.title("Memorization Comparison: Normal vs DP-LoRA")
        plt.tight_layout()
        bar_path = output_dir / "comparison_bar.png"
        plt.savefig(bar_path, dpi=150)
        plt.close()
        print(f"Saved bar plot to: {bar_path}")
    else:
        print("[WARNING] Skipping bar plot because normal or DP-LoRA metrics are unavailable.")

    if sigma_metrics:
        sigma_metrics = sorted(sigma_metrics, key=lambda x: x[0])
        sigmas = np.array([sigma for sigma, _ in sigma_metrics], dtype=float)
        avg_duplicates = np.array([m["avg_duplicates"] for _, m in sigma_metrics], dtype=float)

        plt.figure(figsize=(6, 4))
        plt.plot(sigmas, avg_duplicates, marker="o", color="tab:green")
        plt.xlabel("Sigma")
        plt.ylabel("Average duplicates")
        plt.title("DP-LoRA Privacy Tradeoff")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        sigma_path = output_dir / "sigma_vs_duplicates.png"
        plt.savefig(sigma_path, dpi=150)
        plt.close()
        print(f"Saved sigma plot to: {sigma_path}")
    else:
        print("[INFO] No DP-LoRA sigma sweep files found for sigma-vs-duplicates plot.")


def main() -> None:
    results_dir = Path.cwd()
    plot_dir = results_dir / "plots"

    normal_results, dp_lora_results, dp_sigma_results_raw = load_results(results_dir)

    normal_metrics = compute_metrics(normal_results)
    dp_metrics = compute_metrics(dp_lora_results)

    sigma_metrics: List[Tuple[float, Dict[str, float]]] = []
    for sigma, result, path in dp_sigma_results_raw:
        metrics = compute_metrics(result)
        if metrics is None:
            print(f"[WARNING] Skipping malformed sigma file: {path.name}")
            continue
        sigma_metrics.append((sigma, metrics))

    print("\n=== RESULTS COMPARISON ===\n")

    if normal_metrics is not None:
        print("Normal Model:")
        print(f"  Avg duplicates: {normal_metrics['avg_duplicates']:.4f}")
        print(f"  Total embeddings: {normal_metrics['total_embeddings']}")
    else:
        print("Normal Model:")
        print("  Metrics unavailable")

    print()

    if dp_metrics is not None:
        print("DP-LoRA Model:")
        print(f"  Avg duplicates: {dp_metrics['avg_duplicates']:.4f}")
        print(f"  Total embeddings: {dp_metrics['total_embeddings']}")

        if normal_metrics is not None and normal_metrics["avg_duplicates"] != 0:
            reduction = normal_metrics["avg_duplicates"] - dp_metrics["avg_duplicates"]
            reduction_percent = (reduction / normal_metrics["avg_duplicates"]) * 100.0
            print(f"  Reduction: {reduction:.4f}")
            print(f"  Reduction (%): {reduction_percent:.2f}%")
        elif normal_metrics is not None:
            print("  Reduction: undefined (normal avg duplicates is 0)")
            print("  Reduction (%): undefined")
    else:
        print("DP-LoRA Model:")
        print("  Metrics unavailable")

    plot_results(normal_metrics, dp_metrics, sigma_metrics, plot_dir)


if __name__ == "__main__":
    main()
