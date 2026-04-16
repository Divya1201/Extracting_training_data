"""
Export URL/caption files from LAION-style metadata for img2dataset.

Use this before build_laion_paper_dataset.py when your metadata contains URLs
but images have not been downloaded yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LAION URL/caption metadata for image downloading."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/laion_urls.csv"))
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--caption-col", default="caption")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--aesthetic-col", default=None)
    parser.add_argument("--min-aesthetic", type=float, default=None)
    parser.add_argument("--width-col", default=None)
    parser.add_argument("--height-col", default=None)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--watermark-col", default=None)
    parser.add_argument("--max-watermark", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _read_table(args.metadata)

    if args.aesthetic_col and args.min_aesthetic is not None:
        df = df[df[args.aesthetic_col] >= args.min_aesthetic]
    if args.width_col and args.height_col:
        df = df[(df[args.width_col] >= args.min_size) & (df[args.height_col] >= args.min_size)]
    if args.watermark_col:
        df = df[df[args.watermark_col] < args.max_watermark]

    output = pd.DataFrame({
        "url": df[args.url_col].astype(str),
        "caption": df[args.caption_col].astype(str),
    })
    if args.id_col:
        output["id"] = df[args.id_col].astype(str)

    output = output[(output["url"].str.len() > 0) & (output["caption"].str.len() > 0)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()

