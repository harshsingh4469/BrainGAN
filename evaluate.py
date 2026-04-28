import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gan_model import ConditionalGenerator

# --- Load EEG data from a folder ---
def load_eeg_data(data_dir):
    files  = sorted([os.path.join(data_dir, f)
             for f in os.listdir(data_dir) if f.endswith('.npy')])
    X, y = [], []
    for f in files:
        eeg   = np.load(f).flatten()
        label = int(f.split('label')[1].split('.')[0])
        X.append(eeg)
        y.append(label)
    return np.array(X), np.array(y)

def evaluate_robustness():
    print("=" * 50)
    print("BrainGAN Evaluation: Robustness Test")
    print("=" * 50)

    # --- Test 1: Baseline (real data only, small dataset) ---
    print("\n[1] Baseline: Training with only 20 real samples...")
    X_real, y_real = load_eeg_data("data/real")
    X_small = X_real[:20]
    y_small = y_real[:20]
    X_tr, X_te, y_tr, y_te = train_test_split(X_real, y_real, 
                               test_size=0.2, random_state=42)
    X_tr_small = X_real[:16]
    y_tr_small = y_real[:16]

    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_baseline.fit(X_tr_small, y_tr_small)
    acc_baseline = accuracy_score(y_te, clf_baseline.predict(X_te))
    print(f"    Baseline Accuracy: {acc_baseline*100:.1f}%")

    # --- Test 2: With GAN augmentation ---
    print("\n[2] With GAN augmented data...")
    X_syn, y_syn = load_eeg_data("data/synthetic")
    X_aug = np.concatenate([X_tr_small, X_syn])
    y_aug = np.concatenate([y_tr_small, y_syn])

    clf_gan = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_gan.fit(X_aug, y_aug)
    acc_gan = accuracy_score(y_te, clf_gan.predict(X_te))
    print(f"    GAN Augmented Accuracy: {acc_gan*100:.1f}%")

    # --- Test 3: With Diffusion augmentation ---
    print("\n[3] With Diffusion augmented data...")
    diff_files = sorted([os.path.join("data/synthetic_diffusion", f)
                 for f in os.listdir("data/synthetic_diffusion") 
                 if f.endswith('.npy')])
    X_diff = np.array([np.load(f).flatten() for f in diff_files])
    y_diff = np.array([i % 2 for i in range(len(diff_files))])

    X_aug2 = np.concatenate([X_tr_small, X_diff])
    y_aug2 = np.concatenate([y_tr_small, y_diff])

    clf_diff = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_diff.fit(X_aug2, y_aug2)
    acc_diff = accuracy_score(y_te, clf_diff.predict(X_te))
    print(f"    Diffusion Augmented Accuracy: {acc_diff*100:.1f}%")

    # --- Results Summary ---
    improvement_gan  = ((acc_gan  - acc_baseline) / acc_baseline) * 100
    improvement_diff = ((acc_diff - acc_baseline) / acc_baseline) * 100
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Baseline Accuracy:             {acc_baseline*100:.1f}%")
    print(f"GAN Augmented Accuracy:        {acc_gan*100:.1f}%  (+{improvement_gan:.1f}%)")
    print(f"Diffusion Augmented Accuracy:  {acc_diff*100:.1f}%  (+{improvement_diff:.1f}%)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    methods = ['Baseline\n(real only)', 'GAN\nAugmented', 'Diffusion\nAugmented']
    accs    = [acc_baseline*100, acc_gan*100, acc_diff*100]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']
    axes[0].bar(methods, accs, color=colors, width=0.5)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('BrainGAN: Classification Robustness')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

    # EEG sample comparison
    real_sample = np.load("data/real/eeg_000_label0.npy")
    syn_files   = sorted(os.listdir("data/synthetic"))
    syn_sample  = np.load(f"data/synthetic/{syn_files[0]}")
    axes[1].plot(real_sample[0], label='Real EEG',      alpha=0.8, color='#e74c3c')
    axes[1].plot(syn_sample[0],  label='GAN Synthetic', alpha=0.8, color='#2ecc71')
    axes[1].set_title('Real vs GAN Synthetic EEG Signal')
    axes[1].set_xlabel('Timepoints')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/braingan_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved as results/braingan_results.png!")

if __name__ == "__main__":
    evaluate_robustness()