import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from gan_model import ConditionalGenerator, ConditionalDiscriminator

# --- Dataset ---
class EEGDataset(Dataset):
    def __init__(self, data_dir):
        self.files  = sorted([os.path.join(data_dir, f) 
                      for f in os.listdir(data_dir) if f.endswith('.npy')])
        print(f"Found {len(self.files)} EEG samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        eeg   = torch.FloatTensor(np.load(fname))
        label = int(fname.split('label')[1].split('.')[0])
        return eeg, label

# --- Config ---
DEVICE    = "cpu"
EPOCHS    = 50
BATCH     = 16
NOISE_DIM = 100
LR        = 0.0002

def train_gan():
    dataset = EEGDataset("data/real")
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    G = ConditionalGenerator(NOISE_DIM).to(DEVICE)
    D = ConditionalDiscriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print("Starting GAN training...\n")

    for epoch in range(EPOCHS):
        d_losses, g_losses = [], []

        for real_eeg, labels in loader:
            real_eeg = real_eeg.to(DEVICE)
            labels   = labels.to(DEVICE)
            batch    = real_eeg.size(0)

            real_labels = torch.ones(batch,  1).to(DEVICE)
            fake_labels = torch.zeros(batch, 1).to(DEVICE)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            d_real = D(real_eeg, labels)
            d_loss_real = criterion(d_real, real_labels)

            z        = torch.randn(batch, NOISE_DIM).to(DEVICE)
            fake_eeg = G(z, labels)
            d_fake   = D(fake_eeg.detach(), labels)
            d_loss_fake = criterion(d_fake, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            d_fake  = D(fake_eeg, labels)
            g_loss  = criterion(d_fake, real_labels)
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | D Loss: {np.mean(d_losses):.4f} | G Loss: {np.mean(g_losses):.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(G.state_dict(), "models/generator.pt")
    torch.save(D.state_dict(), "models/discriminator.pt")
    print("\nGAN training complete! Models saved in models/")

    # Generate synthetic EEG samples
    print("\nGenerating synthetic EEG samples...")
    os.makedirs("data/synthetic", exist_ok=True)
    G.eval()
    with torch.no_grad():
        for cls in range(2):
            for i in range(50):
                z     = torch.randn(1, NOISE_DIM)
                label = torch.LongTensor([cls])
                fake  = G(z, label).squeeze(0).numpy()
                np.save(f"data/synthetic/eeg_{cls}_{i:03d}_label{cls}.npy", fake)
    print("Generated 100 synthetic EEG samples in data/synthetic/")

if __name__ == "__main__":
    train_gan()