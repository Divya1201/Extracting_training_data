from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline, PNDMScheduler


# ---------------------------------------------------------------------------
# Load Stable Diffusion (v1.4 + PLMS sampler as in paper §4.2)
# ---------------------------------------------------------------------------

def load_stable_diffusion(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    use_plms: bool = True,
    device: Optional[str] = None,
) -> StableDiffusionPipeline:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if use_plms:
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.safety_checker = None
    return pipe


# ---------------------------------------------------------------------------
# Generate 500 images per prompt (paper §4.2.1)
# ---------------------------------------------------------------------------

def generate_images_for_prompt(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_dir: Path,
    num_images: int = 500,   #50
    image_size: int = 512,
    guidance_scale: float = 7.5,
    batch_size: int = 8,
) -> List[Path]:

    safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:80]
    prompt_dir = output_dir / safe_prompt
    prompt_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    generated = 0

    while generated < num_images:
        batch = min(batch_size, num_images - generated)

        # reproducible seeds for each image
        generators = [
            torch.Generator(device=pipe.device).manual_seed(generated + i)
            for i in range(batch)
        ]

        images = pipe(
            [prompt] * batch,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
            generator=generators,
        ).images

        for i, img in enumerate(images):
            out_path = prompt_dir / f"{generated + i:04d}.png"
            img.save(out_path)
            saved.append(out_path)

        generated += batch

        if generated % 50 == 0:
            print(f"  Generated {generated}/{num_images} for '{prompt}'")

    return saved


# ---------------------------------------------------------------------------
# Generate for all prompts (used in §4.2 experiments)
# ---------------------------------------------------------------------------

def generate_all_prompts(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    output_dir: Path,
    num_images_per_prompt: int = 500,   #50
    batch_size=8,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt}")
        generate_images_for_prompt(
            pipe,
            prompt,
            output_dir,
            num_images=num_images_per_prompt,
            batch_size=batch_size,
        )

    print("\n All images generated.")


