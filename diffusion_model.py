import torch
import torch.nn as nn
import numpy as np

# --- Simple Diffusion Model for EEG ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EEGDiffusionBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Linear(time_dim, channels)
        self.act = nn.GELU()

    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class EEGDiffusionModel(nn.Module):
    """
    Denoising Diffusion Model for EEG generation.
    Learns to remove noise from EEG signals step by step.
    """
    def __init__(self, n_channels=64, seq_len=256, time_dim=128, n_steps=1000):
        super().__init__()
        self.n_steps  = n_steps
        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Encoder (downsampling)
        self.enc1 = nn.Conv1d(n_channels, 128, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Middle blocks
        self.mid1 = EEGDiffusionBlock(256, time_dim)
        self.mid2 = EEGDiffusionBlock(256, time_dim)

        # Decoder (upsampling)
        self.dec2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(128, n_channels, kernel_size=3, padding=1)

        self.act  = nn.GELU()
        self.norm = nn.GroupNorm(8, 256)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Encode
        h1 = self.act(self.enc1(x))
        h2 = self.act(self.enc2(h1))

        # Middle
        h  = self.mid1(h2, t_emb)
        h  = self.mid2(h,  t_emb)

        # Decode
        h  = self.act(self.dec2(h))
        h  = self.dec1(h)
        return h

    def add_noise(self, x, t):
        """Add noise to EEG at timestep t (forward diffusion)"""
        noise     = torch.randn_like(x)
        alpha     = 1 - (t.float() / self.n_steps).view(-1, 1, 1)
        noisy_x   = alpha * x + (1 - alpha) * noise
        return noisy_x, noise

    @torch.no_grad()
    def sample(self, n_samples, n_channels=64, seq_len=256, device="cpu"):
        """Generate new EEG samples from pure noise"""
        x = torch.randn(n_samples, n_channels, seq_len).to(device)
        for t in reversed(range(0, self.n_steps, self.n_steps // 50)):
            t_batch   = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted = self(x, t_batch)
            alpha     = 1 - t / self.n_steps
            x         = (x - (1 - alpha) * predicted) / alpha
            x         = torch.clamp(x, -3, 3)
        return x