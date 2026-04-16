from PIL import Image
from pathlib import Path

input_dir = Path("laion_subset")
output_dir = Path("data/laion_images")
output_dir.mkdir(parents=True, exist_ok=True)

i = 0

print("Processing images...")

for path in input_dir.rglob("*.*"):
    try:
        img = Image.open(path).convert("RGB").resize((512, 512))
        img.save(output_dir / f"img_{i:05d}.png")
        i += 1
    except:
        continue

print(f"Saved {i} images to data/laion_images")
