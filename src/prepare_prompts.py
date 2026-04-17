from pathlib import Path
import argparse
import csv
import json
import pickle
import random

# ==============================
# BASE PROMPTS 
# ==============================

BASE_PROMPTS = [
    "a stock photo of a smiling woman looking at the camera",
    "a professional portrait of a smiling woman",
    "a close up photo of a smiling woman",

    "a product image of white sneakers on plain background",
    "white sneakers product photography",

    "a professional headshot of a man in a suit studio lighting",
    "a corporate portrait of a man in formal attire",

    "a close-up photo of a dog sitting on grass",
    "a dog portrait outdoors natural light",
    "a pet dog sitting in a park"
]

# ==============================
# SIMULATE DUPLICATION (CRITICAL)
# ==============================

NUM_TOTAL_PROMPTS = 100   #30   # simulate "many captions"
HIGH_DUP_FACTOR = 20     #10      # strong duplication


def _load_captions_csv(path: Path, caption_column: str) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row[caption_column].strip() for row in reader if row.get(caption_column)]


def _load_captions_jsonl(path: Path, caption_key: str) -> list[str]:
    captions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            caption = str(item.get(caption_key, "")).strip()
            if caption:
                captions.append(caption)
    return captions


def _load_duplicate_counts(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def prepare_simulated_prompts(total_prompts: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts = []

    # Strongly duplicated prompts (like LAION top duplicates)
    for p in BASE_PROMPTS:
        prompts.extend([p] * HIGH_DUP_FACTOR)

    # Add some random variation (optional realism)
    rng.shuffle(prompts)

    # Limit total prompts
    return prompts[:total_prompts]


def prepare_ranked_prompts(
    captions: list[str],
    duplicate_counts,
    top_k: int,
) -> list[str]:
    """
    Paper-style prompt selection: sort training examples by duplicate count and
    keep captions of the most duplicated examples.

    ``duplicate_counts`` may be either:
      - dict[int, count], where int indexes into ``captions``
      - dict[str, count], where str is itself a caption

    If captions already came from scripts/build_laion_paper_dataset.py, they
    are already duplicate-ranked. In that case, run this script without
    --duplicate-counts and pass --captions-csv.
    """
    sorted_items = sorted(duplicate_counts.items(), key=lambda item: item[1], reverse=True)
    prompts = []
    for key, _ in sorted_items:
        if isinstance(key, int) and 0 <= key < len(captions):
            prompts.append(captions[key])
        elif isinstance(key, str):
            try:
                idx = int(key)
            except ValueError:
                prompts.append(key)
            else:
                if 0 <= idx < len(captions):
                    prompts.append(captions[idx])
        if len(prompts) >= top_k:
            break
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare prompts for the Carlini et al. duplicated-caption attack."
    )
    parser.add_argument("--output", type=Path, default=Path("prompts.txt"))
    parser.add_argument("--captions-csv", type=Path, default=None)
    parser.add_argument("--captions-jsonl", type=Path, default=None)
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--caption-key", default="caption")
    parser.add_argument("--duplicate-counts", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=350_000)
    parser.add_argument("--simulate-count", type=int, default=NUM_TOTAL_PROMPTS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.duplicate_counts and (args.captions_csv or args.captions_jsonl):
        if args.captions_csv:
            captions = _load_captions_csv(args.captions_csv, args.caption_column)
        else:
            captions = _load_captions_jsonl(args.captions_jsonl, args.caption_key)
        duplicate_counts = _load_duplicate_counts(args.duplicate_counts)
        prompts = prepare_ranked_prompts(captions, duplicate_counts, args.top_k)
        mode = "duplicate-ranked captions"
    elif args.captions_csv or args.captions_jsonl:
        if args.captions_csv:
            prompts = _load_captions_csv(args.captions_csv, args.caption_column)[:args.top_k]
        else:
            prompts = _load_captions_jsonl(args.captions_jsonl, args.caption_key)[:args.top_k]
        mode = "pre-ranked captions"
    else:
        prompts = prepare_simulated_prompts(args.simulate_count, args.seed)
        mode = "simulated duplicate prompts"

    # Save
    output_file = args.output
    with open(output_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"Saved {len(prompts)} prompts to {output_file}")
    print(f"Mode: {mode}")

if __name__ == "__main__":
    main()
