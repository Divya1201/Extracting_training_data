from pathlib import Path
import shutil
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from dataset import generate_cifar10_splits

# ==============================
# CONFIG
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 300        #3
NUM_MODELS = 16     #2
MAX_IMAGES_PER_MODEL = None
CLASS_CONDITIONAL = True

SAVE_DIR = Path("models")
SAVE_DIR.mkdir(exist_ok=True)

# ==============================
# DATASET
# ==============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

dataset = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# IMPORTANT: SAME SPLITS AS MEMBERSHIP EVAL
splits = generate_cifar10_splits(n_models=NUM_MODELS)

# ==============================
# TRAIN FUNCTION
# ==============================

def train_model(model_idx, member_indices):
    print(f"\n Training model {model_idx}...")

    if MAX_IMAGES_PER_MODEL is not None:
        member_indices = member_indices[:MAX_IMAGES_PER_MODEL]

    subset = Subset(dataset, member_indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    # Diffusion model (DDPM UNet)
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),  # (64, 128, 128, 256)
        num_class_embeds=10 if CLASS_CONDITIONAL else None,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(DEVICE)

    scheduler = DDPMScheduler(num_train_timesteps=1000)    #500

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ==============================
    # TRAIN LOOP
    # ==============================

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (images.size(0),), device=DEVICE
            )

            noisy_images = scheduler.add_noise(images, noise, timesteps)

            if CLASS_CONDITIONAL:
                noise_pred = model(noisy_images, timesteps, class_labels=labels).sample
            else:
                noise_pred = model(noisy_images, timesteps).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================

    model_dir = SAVE_DIR / f"model_{model_idx}"
    model_dir.mkdir(exist_ok=True)

    pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    pipeline.save_pretrained(model_dir)

    print(f" Saved model {model_idx} → {model_dir}")


# ==============================
# MAIN
# ==============================

def main():
    print(" Starting training of shadow models...")

    for i, (members, _) in enumerate(splits):
        train_model(i, members)

    print("\n All models trained successfully!")


if __name__ == "__main__":
    main()
