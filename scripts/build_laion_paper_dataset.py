"""
Build the duplicated-caption dataset used by the Stable Diffusion extraction attack.

This script implements the data-selection stage from Carlini et al. §4.2:

  1. Start from image/caption metadata for the Stable-Diffusion training subset.
  2. Embed every candidate image with CLIP ViT-B/32.
  3. Count near-duplicates by all-pairs CLIP cosine similarity.
  4. Select the most duplicated examples.
  5. Write prompts.txt, captions.csv, duplicate_counts.pkl, and urls.txt.

The paper ran this at LAION scale. This implementation intentionally does the
same conceptual work locally; it is expensive by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_utils import get_clip_embedding


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class MetadataRow:
    row_id: str
    caption: str
    image_path: Optional[Path]
    url: str


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported metadata format: {path}")


def _resolve_image_path(value: str, image_root: Optional[Path]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    return path if path.suffix.lower() in IMAGE_SUFFIXES else None


def _iter_metadata_rows(
    metadata: pd.DataFrame,
    caption_col: str,
    image_col: str,
    url_col: Optional[str],
    id_col: Optional[str],
    image_root: Optional[Path],
) -> Iterator[MetadataRow]:
    for idx, row in metadata.iterrows():
        caption = str(row.get(caption_col, "")).strip()
        if not caption:
            continue

        raw_image_path = str(row.get(image_col, "")).strip()
        image_path = _resolve_image_path(raw_image_path, image_root)
        if image_path is None or not image_path.exists():
            continue

        url = str(row.get(url_col, "")).strip() if url_col else ""
        row_id = str(row.get(id_col, idx)).strip() if id_col else str(idx)
        yield MetadataRow(row_id=row_id, caption=caption, image_path=image_path, url=url)


def _embed_rows(
    rows: list[MetadataRow],
    cache_path: Path,
    resume: bool,
) -> np.ndarray:
    if resume and cache_path.exists():
        print(f"Loading cached CLIP embeddings: {cache_path}")
        return np.load(cache_path)

    embeddings = np.zeros((len(rows), 512), dtype=np.float32)
    for i, row in enumerate(rows):
        embeddings[i] = get_clip_embedding(row.image_path)
        if (i + 1) % 100 == 0:
            print(f"Embedded {i + 1}/{len(rows)} images")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def _duplicate_counts_all_pairs(
    embeddings: np.ndarray,
    cosine_threshold: float,
    block_size: int,
) -> np.ndarray:
    """
    Count CLIP near-duplicates for every sample.

    This computes the exact blockwise all-pairs cosine threshold graph. CLIP
    embeddings are unit-normalized by clip_utils, so dot product is cosine.
    """
    n = embeddings.shape[0]
    counts = np.zeros(n, dtype=np.int64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = embeddings[start:end] @ embeddings.T
        sims[:, start:end] -= np.eye(end - start, dtype=sims.dtype)
        block_hits = sims >= cosine_threshold
        counts[start:end] += block_hits.sum(axis=1)
        print(f"Counted duplicate similarities for {end}/{n}")

    return counts


def _write_outputs(
    rows: list[MetadataRow],
    duplicate_counts: np.ndarray,
    output_dir: Path,
    top_k: int,
    copy_images: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = np.argsort(-duplicate_counts)[:top_k]

    prompts_path = output_dir / "prompts.txt"
    captions_path = output_dir / "captions.csv"
    urls_path = output_dir / "urls.txt"
    duplicate_counts_path = output_dir / "duplicate_counts.pkl"
    reference_dir = output_dir / "reference_images"

    if copy_images:
        reference_dir.mkdir(exist_ok=True)

    with prompts_path.open("w", encoding="utf-8") as prompts_file, \
            captions_path.open("w", newline="", encoding="utf-8") as captions_file, \
            urls_path.open("w", encoding="utf-8") as urls_file:
        writer = csv.writer(captions_file)
        writer.writerow([
            "rank",
            "row_id",
            "caption",
            "image_path",
            "url",
            "duplicate_count",
            "reference_path",
        ])

        selected_counts = {}
        for rank, idx in enumerate(selected_indices):
            row = rows[int(idx)]
            reference_path = row.image_path
            if copy_images:
                out_name = f"{rank:06d}_{row.image_path.name}"
                reference_path = reference_dir / out_name
                shutil.copy2(row.image_path, reference_path)

            prompts_file.write(row.caption + "\n")
            if row.url:
                urls_file.write(row.url + "\n")
            writer.writerow([
                rank,
                row.row_id,
                row.caption,
                str(row.image_path),
                row.url,
                int(duplicate_counts[idx]),
                str(reference_path),
            ])
            selected_counts[str(reference_path)] = int(duplicate_counts[idx])

    with duplicate_counts_path.open("wb") as f:
        pickle.dump(selected_counts, f)

    print(f"Wrote prompts: {prompts_path}")
    print(f"Wrote selected captions: {captions_path}")
    print(f"Wrote duplicate counts: {duplicate_counts_path}")
    if copy_images:
        print(f"Wrote reference images: {reference_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper-style duplicated LAION prompt/reference subset."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/laion_paper_subset"))
    parser.add_argument("--caption-col", default="caption")
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--top-k", type=int, default=350_000)
    parser.add_argument("--cosine-threshold", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy selected reference images into output-dir/reference_images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = _read_table(args.metadata)

    rows = list(_iter_metadata_rows(
        metadata=metadata,
        caption_col=args.caption_col,
        image_col=args.image_col,
        url_col=args.url_col,
        id_col=args.id_col,
        image_root=args.image_root,
    ))
    if not rows:
        raise ValueError("No usable rows found. Check caption/image columns and image paths.")

    print(f"Loaded {len(rows)} image-caption rows")
    cache_path = args.embedding_cache or (args.output_dir / "clip_embeddings.npy")
    embeddings = _embed_rows(rows, cache_path=cache_path, resume=args.resume)
    duplicate_counts = _duplicate_counts_all_pairs(
        embeddings=embeddings,
        cosine_threshold=args.cosine_threshold,
        block_size=args.block_size,
    )
    _write_outputs(
        rows=rows,
        duplicate_counts=duplicate_counts,
        output_dir=args.output_dir,
        top_k=min(args.top_k, len(rows)),
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()

