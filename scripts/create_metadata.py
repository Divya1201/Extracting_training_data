from pathlib import Path
import pandas as pd

image_dir = Path("data/laion_images")

rows = []

for img_path in image_dir.rglob("*"):
    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        rows.append({
            "image_path": str(img_path),
            "caption": img_path.stem.replace("_", " ")
        })

df = pd.DataFrame(rows)
df.to_csv("data/laion_metadata.csv", index=False)

print(f" Created metadata with {len(df)} images")
