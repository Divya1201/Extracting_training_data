from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffusion import (
    load_stable_diffusion,
    generate_all_prompts,
)

# ==============================
# CONFIG 
# ==============================

OUTPUT_DIR = Path("outputs/generated")

def load_prompts(file="prompts.txt"):
    if not Path(file).exists():
        raise FileNotFoundError(
            " prompts.txt not found.\n"
            " Run: python prepare_prompts.py"
        )

    with open(file, "r") as f:
        return [line.strip() for line in f.readlines()]

PROMPTS = load_prompts()

IMAGES_PER_PROMPT = 500  #50   # REQUIRED (paper: 500 generations)

BATCH_SIZE =  8  #4            # GPU: 6–8 | CPU: 2–4

# ==============================
# MAIN
# ==============================

def main():
    print("Loading Stable Diffusion v1.4 (PLMS)...")

    pipe = load_stable_diffusion()

    device = pipe.device
    print(f" Running on: {device}")

    print("\n Generating images (500 per prompt)...")

    generate_all_prompts(
        pipe=pipe,
        prompts=PROMPTS,
        output_dir=OUTPUT_DIR,
        num_images_per_prompt=IMAGES_PER_PROMPT,
        batch_size=BATCH_SIZE,
    )

    print("\n Generation complete!")


if __name__ == "__main__":
    main()

