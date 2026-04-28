# BrainGAN: Generative Models for Neural Data Augmentation

A deep learning project that uses GANs and Diffusion Models to generate 
synthetic EEG datasets, improving BCI classification robustness in low-data regimes.

## What it does
- Trains a **Conditional GAN** to generate class-specific synthetic EEG signals
- Trains a **Diffusion Model** to generate realistic EEG via denoising
- Evaluates how synthetic data improves BCI classification accuracy

## Results
| Method | Accuracy |
|--------|----------|
| Baseline (real data only) | 100.0% |
| GAN Augmented | 85.0% |
| Diffusion Augmented | 95.0% |

## Architecture
- **ConditionalGenerator** — generates class-specific EEG from noise
- **ConditionalDiscriminator** — distinguishes real vs fake EEG
- **EEGDiffusionModel** — denoising diffusion model for EEG generation

## Setup & Usage

### Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate real EEG data
```bash
python3 generate_eeg_data.py
```

### Train GAN
```bash
python3 train_gan.py
```

### Train Diffusion Model
```bash
python3 train_diffusion.py
```

### Evaluate robustness
```bash
python3 evaluate.py
```

## Tech Stack
- PyTorch
- Hugging Face Diffusers
- Hugging Face Transformers
- scikit-learn
- Matplotlib

## Author
Harsh Singh — [GitHub](https://github.com/harshsingh4469)
