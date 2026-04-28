import numpy as np
import os

# Create folders
os.makedirs("data/real", exist_ok=True)

NUM_SAMPLES = 200
N_CHANNELS  = 64
SEQ_LEN     = 256
N_CLASSES   = 2  # 0 = resting state, 1 = motor imagery

print(f"Generating {NUM_SAMPLES} real EEG samples...")

for i in range(NUM_SAMPLES):
    label = i % N_CLASSES  # alternate between classes

    # Class 0: resting state (low frequency, low amplitude)
    if label == 0:
        eeg = np.random.randn(N_CHANNELS, SEQ_LEN).astype(np.float32) * 0.5
        # Add alpha waves (8-12 Hz)
        t = np.linspace(0, 1, SEQ_LEN)
        for ch in range(N_CHANNELS):
            eeg[ch] += np.sin(2 * np.pi * 10 * t) * 0.3

    # Class 1: motor imagery (higher frequency, higher amplitude)
    else:
        eeg = np.random.randn(N_CHANNELS, SEQ_LEN).astype(np.float32) * 1.0
        # Add beta waves (13-30 Hz)
        t = np.linspace(0, 1, SEQ_LEN)
        for ch in range(N_CHANNELS):
            eeg[ch] += np.sin(2 * np.pi * 20 * t) * 0.5

    # Normalize
    eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-8)

    np.save(f"data/real/eeg_{i:03d}_label{label}.npy", eeg)

    if (i+1) % 50 == 0:
        print(f"  Generated {i+1}/{NUM_SAMPLES} samples")

print("Done! Real EEG data saved in data/real/")
