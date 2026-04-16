"""
Membership Inference Attacks on Diffusion Models (§5.2)
Carlini et al. (2023)

Two attacks:
  1. Loss threshold attack (§5.2, Yeom et al.) — predict 'member' if
     diffusion loss at timestep t=100 is below threshold τ.
  2. LiRA (Carlini et al. 2022) — train 16 shadow models on random 50%
     splits; compare in-model vs. out-model loss distributions.

Also implements the augmentation improvements (§5.2.2):
  - Average loss over multiple noise samples (reduces variance).
  - Average loss over horizontal flips of the image.

Requires:
  - Trained diffusion model(s) accessible via HuggingFace diffusers or
    the OpenAI improved-diffusion codebase.
  - For LiRA: 16 shadow models (expensive but tractable on CIFAR-10).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Diffusion loss at a fixed timestep (§5.2)
# ---------------------------------------------------------------------------

def compute_diffusion_loss(
    model,
    noise_scheduler,
    image_tensor: torch.Tensor,
    timestep: int = 100,
    n_samples: int = 1,
    use_horizontal_flip: bool = False,
    device: str = "cpu",
    class_label: Optional[int] = None,
) -> float:
    """
    Compute the reconstruction loss L(x, t, ε) at a fixed noise timestep t.

    Paper (§5.2): "We find that evaluating L(·,t,·) at t ∈ [50, 300]
    produces the best results." Default t=100.

    Augmentations (§5.2.2):
      - n_samples > 1: average over multiple noise draws (reduces variance).
      - use_horizontal_flip: also average over the horizontally-flipped image.

    Returns the mean scalar loss.
    """
    model.eval()
    losses = []
    images_to_eval = [image_tensor]
    if use_horizontal_flip:
        images_to_eval.append(torch.flip(image_tensor, dims=[-1]))

    batch_size = image_tensor.shape[0]
    t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
    class_labels = None
    if class_label is not None:
        class_labels = torch.full((batch_size,), class_label, device=device, dtype=torch.long)

    with torch.no_grad():
        for img in images_to_eval:
            img = img.to(device)
            for _ in range(n_samples):
                noise = torch.randn_like(img)
                noisy = noise_scheduler.add_noise(img, noise, t)
                if class_labels is None:
                    predicted_noise = model(noisy, t).sample
                else:
                    predicted_noise = model(noisy, t, class_labels=class_labels).sample
                loss = F.mse_loss(predicted_noise, noise, reduction="mean")
                losses.append(loss.item())

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Loss threshold attack (§5.2 — Yeom et al.)
# ---------------------------------------------------------------------------

def loss_threshold_attack(
    member_losses: List[float],
    nonmember_losses: List[float],
    tau: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Simple loss threshold attack: predict 'member' if loss < τ.

    If τ is None, find the threshold that maximises balanced accuracy.

    Returns (threshold, TPR, FPR).
    """
    all_losses = member_losses + nonmember_losses
    labels = [1] * len(member_losses) + [0] * len(nonmember_losses)

    if tau is None:
        # Search over candidate thresholds
        best_acc, best_tau = 0.0, 0.0
        for candidate_tau in sorted(set(all_losses)):
            preds = [1 if l < candidate_tau else 0 for l in all_losses]
            acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
            if acc > best_acc:
                best_acc = acc
                best_tau = candidate_tau
        tau = best_tau

    preds = [1 if l < tau else 0 for l in all_losses]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))

    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return float(tau), float(tpr), float(fpr)


# ---------------------------------------------------------------------------
# LiRA — Likelihood Ratio Attack (§5.2, Carlini et al. 2022)
# ---------------------------------------------------------------------------

def fit_gaussian(losses: List[float]) -> Tuple[float, float]:
    """Fit a Gaussian N(μ, σ²) to a list of scalar losses."""
    arr = np.array(losses, dtype=np.float64)
    return float(arr.mean()), float(arr.std() + 1e-8)


def lira_score(
    loss_star: float,
    in_losses: List[float],
    out_losses: List[float],
) -> float:
    """
    LiRA membership score (§5.2).

    Fit Gaussians to in-model losses (N_IN) and out-model losses (N_OUT).
    Return log Pr[l* | N_IN] − log Pr[l* | N_OUT].
    Positive → predict 'member'.

    Paper: "measure whether Pr[l* | N_IN] > Pr[l* | N_OUT]."
    """
    def log_gaussian(x: float, mu: float, sigma: float) -> float:
        return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))

    mu_in, sigma_in = fit_gaussian(in_losses)
    mu_out, sigma_out = fit_gaussian(out_losses)

    log_p_in = log_gaussian(loss_star, mu_in, sigma_in)
    log_p_out = log_gaussian(loss_star, mu_out, sigma_out)
    return log_p_in - log_p_out


class LiRAAttack:
    """
    LiRA wrapper for diffusion models (§5.2).

    Usage:
        attack = LiRAAttack(shadow_model_losses_in, shadow_model_losses_out)
        score = attack.score(target_loss)
        # score > 0 → predict member
    """

    def __init__(
        self,
        in_losses: List[float],   # losses from shadow models that DID train on x
        out_losses: List[float],  # losses from shadow models that did NOT train on x
    ) -> None:
        self.in_losses = in_losses
        self.out_losses = out_losses

    def score(self, loss_star: float) -> float:
        """Return log-likelihood ratio score. Positive → predict member."""
        return lira_score(loss_star, self.in_losses, self.out_losses)

    def predict(self, loss_star: float) -> int:
        """Return 1 (member) or 0 (non-member)."""
        return int(self.score(loss_star) > 0)


# ---------------------------------------------------------------------------
# Optimal timestep search (§5.2)
# ---------------------------------------------------------------------------

def find_optimal_timestep(
    model,
    noise_scheduler,
    member_images: List[torch.Tensor],
    nonmember_images: List[torch.Tensor],
    timesteps: Optional[List[int]] = None,
    device: str = "cpu",
) -> int:
    """
    Exhaustively search for the timestep t ∈ [1, T] that maximises
    TPR@FPR=1% for the loss-threshold attack.

    Paper (§5.2, Figure 9): best results at t ∈ [50, 300]; default t=100.

    This is expensive. Pre-compute and cache results.
    """
    if timesteps is None:
        timesteps = list(range(10, 501, 10))    #[50,100,200]   # sample t: 10, 20, …, 500

    best_t, best_tpr = 100, 0.0
    for t in timesteps:
        m_losses = [compute_diffusion_loss(model, noise_scheduler, img, t, device=device)
                    for img in member_images]
        nm_losses = [compute_diffusion_loss(model, noise_scheduler, img, t, device=device)
                     for img in nonmember_images]

        # TPR @ FPR=1%
        threshold_at_fpr1 = np.percentile(nm_losses, 1)  # top 1% of non-member losses
        tpr = np.mean([l < threshold_at_fpr1 for l in m_losses])

        print(f"  t={t:4d}  TPR@FPR=1%: {tpr:.3f}")
        if tpr > best_tpr:
            best_tpr = tpr
            best_t = t

    print(f" Best timestep: t={best_t}  TPR@FPR=1%={best_tpr:.3f}")
    return best_t
