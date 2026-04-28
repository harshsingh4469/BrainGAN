import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from diffusion_model import EEGDiffusionModel

# --- Dataset ---
class EEGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([os.path.join(data_dir, f)
                     for f in os.listdir(data_dir) if f.endswith('.npy')])
        print(f"Found {len(self.files)} EEG samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        eeg = torch.FloatTensor(np.load(self.files[idx]))
        return eeg

# --- Config ---
DEVICE   = "cpu"
EPOCHS   = 30
BATCH    = 16
LR       = 1e-4
N_STEPS  = 100  # reduced for CPU speed

def train_diffusion():
    dataset = EEGDataset("data/real")
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model     = EEGDiffusionModel(n_steps=N_STEPS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Starting Diffusion Model training...\n")

    for epoch in range(EPOCHS):
        losses = []

        for eeg in loader:
            eeg = eeg.to(DEVICE)
            b   = eeg.size(0)

            # Random timestep for each sample
            t         = torch.randint(0, N_STEPS, (b,), device=DEVICE)
            noisy_eeg, noise = model.add_noise(eeg, t)

            # Predict the noise
            predicted = model(noisy_eeg, t)
            loss      = torch.nn.MSELoss()(predicted, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {np.mean(losses):.4f}")

        scheduler.step()

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/diffusion_model.pt")
    print("\nDiffusion training complete! Model saved in models/")

    # Generate synthetic EEG samples
    print("\nGenerating synthetic EEG via diffusion...")
    os.makedirs("data/synthetic_diffusion", exist_ok=True)
    model.eval()
    for i in range(50):
        sample = model.sample(1, device=DEVICE)
        np.save(f"data/synthetic_diffusion/eeg_{i:03d}.npy", 
                sample.squeeze(0).numpy())
    print("Generated 50 diffusion EEG samples in data/synthetic_diffusion/")

if __name__ == "__main__":
    train_diffusion()