import torch
import torch.nn as nn

# --- Generator: takes random noise → fake EEG ---
class Generator(nn.Module):
    def __init__(self, noise_dim=100, n_channels=64, seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels

        self.model = nn.Sequential(
            # Input: (batch, noise_dim)
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),

            nn.Linear(1024, n_channels * seq_len),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, self.n_channels, self.seq_len)


# --- Discriminator: takes EEG → real or fake ---
class Discriminator(nn.Module):
    def __init__(self, n_channels=64, seq_len=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


# --- Conditional GAN (knows which class to generate) ---
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, n_classes=2, n_channels=64, seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(noise_dim + n_classes, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),

            nn.Linear(1024, n_channels * seq_len),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.embedding(labels)
        x = torch.cat([z, label_emb], dim=1)
        out = self.model(x)
        return out.view(-1, self.n_channels, self.seq_len)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_classes=2, n_channels=64, seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, seq_len)

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels + 1, 128, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_emb = self.embedding(labels).unsqueeze(1)
        x = torch.cat([x, label_emb], dim=1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)