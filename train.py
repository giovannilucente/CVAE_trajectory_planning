import os
from pathlib import Path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from dataset import Dataset
from tqdm import tqdm
from models.CVAEmod import CVAE
from loss_functions import *

def get_beta(epoch, max_epochs, start=0.0, end=1.0, midpoint=100, slope=10):
    x = (epoch - midpoint) / slope
    beta = start + (end - start) / (1 + np.exp(-x))
    return float(beta)


# -------- Train/Validation Function -------- #
def run_epoch(model, dataloader, criterion, optimizer=None, beta=0.1, scale=10):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_gen_loss = 0.0
    total_ade = 0.0
    loop = tqdm(dataloader, desc="Training" if is_train else "Validating")

    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(loop):
            frames = batch["frames"][:,:3].to(device)                   # (B, C, W, H)
            target = batch["trajectories"][:,0].to(device)          # (B, traj_len, 2)
            reference_path = batch["reference_paths"][:,0].to(device)   # (B, ref_len, 2)
            
            recon, kl = model(target, frames, reference_path)
            loss, recon_loss, kl_loss = criterion(recon, target, kl, beta=beta)

            gen = model.generate(frames, reference_path, batch=frames.shape[0], device=device)
            gen_loss = F.mse_loss(target, gen)
            ade = compute_ADE(gen, target, scale=scale)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_gen_loss +=gen_loss.item()
            total_ade += ade.item()

            avg_loss_so_far = total_loss / (batch_idx + 1)
            avg_recon_loss = total_recon_loss / (batch_idx + 1)
            avg_kl_loss = total_kl_loss / (batch_idx + 1)
            avg_gen_loss = total_gen_loss / (batch_idx + 1)
            avg_ade = total_ade / (batch_idx + 1)

            loop.set_postfix({'rec': avg_recon_loss, 'kl': avg_kl_loss, 'gen': avg_gen_loss, 'ade': avg_ade})

    return total_loss / len(dataloader), avg_ade


# -------- Save Function -------- #
def save_validation_predictions(model, dataloader, device, output_dir,  scale=100.0, num_examples=50):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"[SAVE]")):
            if batch_idx % 100 != 0:
                continue

            frames = batch["frames"][:,:3].to(device)        # (B, T, CAMERAS, CHANNELS, H, W)
            target = batch["trajectories"][:,0].to(device)
            reference_path = batch["reference_paths"][:,0].to(device)

            pred = model.generate(frames, reference_path, batch=frames.shape[0], device=device)

            frames = frames.cpu()
            target = target.cpu()
            pred = pred.cpu()
            reference_path = reference_path.cpu()

            pred = pred * scale
            target = target * scale
            wayp = reference_path * scale

            pred = torch.cumsum(pred, dim=1)
            target = torch.cumsum(target, dim=1)
            wayp = torch.cumsum(wayp, dim=1)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            img = frames[0]

            ax[0].imshow(img.permute(1, 2, 0))
            ax[0].set_title("Input image")
            ax[0].axis("off")

            gt = target[0].numpy()
            pr = pred[0].numpy()
            wayp = wayp[0].numpy()

            ax[1].plot(gt[:, 0], gt[:, 1], "-o", markersize=3, label="Ground truth")
            ax[1].plot(pr[:, 0], pr[:, 1], "-o", markersize=3, label="Generated")
            ax[1].plot(wayp[:, 0], wayp[:, 1], "-o", markersize=3, label="Waypoints")
            ax[1].scatter(0, 0, c="red", marker="*", s=120, label="Vehicle")
            ax[1].arrow(0, 0, 5, 0, width=0.12, head_width=0.5, head_length=0.7, color="red")

            ax[1].set_xlabel("Forward [m]")
            ax[1].set_ylabel("Left [m]")
            ax[1].axis("equal")
            ax[1].grid(True)
            ax[1].legend()
            ax[1].set_title("Trajectory")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{saved:03d}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            saved += 1

            if saved >= num_examples:
                return


if __name__ == "__main__":
    # --- Settings ---
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device: ", device)

    img_dim = 256
    traj_len = 10
    scale = 1.0
    history = 3
    rollout = 1
    dt = 1
    hidden_dim = 64
    latent_dim = 128
    max_samples = None

    transform = Compose([
        Resize(256),      # shortest side = 256
        ToTensor()
    ])

    BASE_DIR = "../../../../mnt/nfs-share/AI_Datasets/_unzipped/world_model"
    TRAIN_DATA_ROOT = os.path.join(BASE_DIR, "train")
    VAL_DATA_ROOT = os.path.join(BASE_DIR, "validation")

    # --- Datasets ---
    train_set = Dataset(TRAIN_DATA_ROOT, image_transform=transform, history=history, rollout=rollout, dt=dt, max_points=traj_len, max_samples=max_samples)
    val_set = Dataset(VAL_DATA_ROOT, image_transform=transform, history=history, rollout=rollout, dt=dt, max_points=traj_len, max_samples=max_samples)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = CVAE(hidden_dim=hidden_dim, input_dim=2, input_len=traj_len, img_size=img_dim, latent_dim=latent_dim)
    #WEIGHTS_PATH = "results/CVAE_64_128_mse_kld_10/best_model.pt" 
    #model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model_name = "CVAE_mod"
    
    criterion = mse_kld
    loss_name = "mse_kld_10"

    beta_start = 0.0
    beta_end = 1.0

    beta_annealer = BetaAnnealer(
        beta_start=beta_start,
        beta_end=beta_end,
        n_steps=epochs,
        schedule='sigmoid'
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    results_dir = Path("results") / f"{model_name}_{loss_name}"
    model_path = results_dir / "best_model.pt"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        beta = beta_annealer.step()
        print(f"\nEpoch {epoch}/{epochs} | Beta: {beta:.4f}")
        train_loss, train_ade = run_epoch(model, train_loader, criterion, optimizer, beta=beta, scale=scale)
        val_loss, val_ade = run_epoch(model, val_loader, criterion, scale=scale)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_validation_predictions(model, val_loader, device, results_dir / f"predictions", scale=scale, num_examples=50)
            torch.save(model.state_dict(), model_path)
            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_ade": float(train_ade),
                "val_ade": float(val_ade),
            }
            with open(results_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved new best model at epoch {epoch}")

    
