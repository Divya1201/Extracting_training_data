import torch
from pathlib import Path
from PIL import Image
import numpy as np

from dataset import generate_cifar10_splits, prepare_cifar10
from membership_inference import (
    compute_diffusion_loss,
    lira_score,
)

from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ==============================
# CONFIG
# ==============================

TIMESTEPS = [1, 10, 50, 100, 200, 300, 500, 800, 1000]
NUM_IMAGES = 500     #50   
NUM_MODELS_TO_USE = 16     #2

MODEL_DIR = Path("models")

# ==============================
# IMAGE LOADING
# ==============================

def load_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    label = int(path.stem.split("_class")[1])
    return tensor, label


# ==============================
# MAIN
# ==============================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(" Preparing dataset...")
    image_dir = prepare_cifar10(Path("data"), num_images=50000)    #1000
    #image_dir = prepare_cifar10(Path("data"))

    print(" Generating CIFAR splits...")
    all_images = sorted(image_dir.glob("img_*.png"))

    splits = generate_cifar10_splits(
        n_models=NUM_MODELS_TO_USE,
        total_images=len(all_images)   
    )
    split_sets = [(set(members), set(nonmembers)) for members, nonmembers in splits]
    
    # ----------------------------
    # Load trained models
    # ----------------------------
    print(" Loading trained shadow models...")

    models = []
    for i in range(NUM_MODELS_TO_USE):
        model_path = MODEL_DIR / f"model_{i}"
        pipe = DDPMPipeline.from_pretrained(model_path)
        pipe = pipe.to(device)
        models.append((pipe.unet, pipe.scheduler))

    # ----------------------------
    # Compute per-example IN / OUT losses
    # ----------------------------
    print(" Computing per-example IN / OUT losses (LiRA-style)...")

    results = []
    for t in TIMESTEPS:
        print(f"\n===== TIMESTEP {t} =====")
        labels_all = []
        scores_all = []

        for img_idx in range(NUM_IMAGES):
            path = all_images[img_idx]
            img, class_label = load_image(path)

            idx = int(path.stem.split("_")[1])
            per_model_losses = []
            per_model_membership = []

            for model_idx, (model, scheduler) in enumerate(models):
                members, nonmembers = split_sets[model_idx]
            
                loss = compute_diffusion_loss(
                    model,
                    scheduler,
                    img,
                    class_label=class_label,
                    timestep=t,
                    device=device,
                )

                if idx in members:
                    per_model_losses.append(loss)
                    per_model_membership.append(1)
                elif idx in nonmembers:
                    per_model_losses.append(loss)
                    per_model_membership.append(0)

            # Paper-style LiRA is per example: for each held-out model loss,
            # fit IN/OUT loss distributions for the same image using the other
            # shadow models.
            for held_out in range(len(per_model_losses)):
                in_losses = [
                    loss for j, loss in enumerate(per_model_losses)
                    if j != held_out and per_model_membership[j] == 1
                ]
                out_losses = [
                    loss for j, loss in enumerate(per_model_losses)
                    if j != held_out and per_model_membership[j] == 0
                ]
                if len(in_losses) < 2 or len(out_losses) < 2:
                    continue
                scores_all.append(lira_score(per_model_losses[held_out], in_losses, out_losses))
                labels_all.append(per_model_membership[held_out])

        labels = np.array(labels_all)
        scores = np.array(scores_all)

        if len(np.unique(labels)) < 2:
            print("Skipping timestep: not enough member/non-member scores.")
            continue

        fpr_lira, tpr_lira, _ = roc_curve(labels, scores)

        idx_1 = np.argmin(np.abs(fpr_lira - 0.01))
        tpr_at_1 = tpr_lira[idx_1]

        print(f"TPR@FPR=1% = {tpr_at_1:.4f}")

        results.append((t, tpr_at_1))

        # ----------------------------
        # Accuracy
        # ----------------------------
        preds = (scores > 0).astype(int)
        accuracy = (preds == labels).mean()

        print(f"LiRA Accuracy: {accuracy:.4f}")

        # ----------------------------
        # Score statistics 
        # ----------------------------
        member_scores = scores[labels == 1]
        nonmember_scores = scores[labels == 0]
        print(f"Mean member score: {member_scores.mean():.4f}")
        print(f"Mean non-member score: {nonmember_scores.mean():.4f}")

       
    timesteps = [r[0] for r in results]
    tprs = [r[1] for r in results]

    plt.figure()
    plt.plot(timesteps, tprs, marker='o')
    plt.xlabel("Diffusion timestep")
    plt.ylabel("TPR @ FPR = 1%")
    plt.title("LiRA Attack vs Timestep (Paper Figure 9)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
