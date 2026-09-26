# GHM_Mamba

Official implementation of:

**GHM-Mamba: A Frequency-Collaborative Multiwavelet Framework with Low-Frequency-Guided Alignment for CT Image Denoising**

## Overview

GHM-Mamba is a CT image denoising framework that integrates the Geronimo–Hardin–Massopust (GHM) multiwavelet transform with Mamba-based feature modeling.

The framework decomposes each input image into 16 GHM coefficient maps organized into low-, mixed-, and high-frequency groups. The proposed architecture combines:

* GHM multiwavelet decomposition and reconstruction
* Mamba-based feature modeling
* Low-frequency-guided alignment
* Frequency-Collaborative Refinement (FCR)
* Band-specific noise prediction
* Full-resolution residual refinement
* Adversarial training

## Dataset

Experiments were conducted using the publicly available **MosMed-L** dataset, derived from MosMedData.

Two PNG slices were selected per patient. The data were split at the patient level with no patient overlap:

| Split      | Patients | Images |
| ---------- | -------: | -----: |
| Training   |      777 |   1554 |
| Validation |      111 |    222 |
| Testing    |      222 |    444 |

Images are single-channel grayscale CT images with a spatial resolution of 352 × 352 pixels.

## Experimental Protocol

The proposed model was trained using:

* Training patch size: `88 × 88`
* Training noise: additive Gaussian noise, σ ∈ [0, 25]
* Validation noise: σ = 25
* Test noise: σ = 25
* Epochs: `30`
* Batch size: `2`
* Optimizer: Adam
* Generator learning rate: `1.5e-4`
* Discriminator learning rate: `1e-4`
* Adam betas: `(0.9, 0.999)`
* EMA decay: `0.999`
* GAN warm-up: `1 epoch`
* Random seed: `123`

The checkpoint with the highest validation PSNR based on EMA weights is used for final evaluation.

## Model Configuration

The full GHM-Mamba configuration uses:

* 16 GHM coefficient maps
* 40 band-feature channels
* 144 feature channels
* 6 Mamba blocks
* Low-frequency-guided alignment
* Frequency-Collaborative Refinement
* Full-resolution refinement
* Adversarial training

## Evaluation

Final evaluation is performed on complete `352 × 352` test images at Gaussian noise level `σ = 25`.

Four image-quality metrics are reported:

* PSNR ↑
* SSIM ↑
* LPIPS ↓
* DISTS ↓

## Results

Across three independent training runs using seeds 123, 456, and 789, GHM-Mamba achieved:

| Metric      |       Mean ± Std |
| ----------- | ---------------: |
| PSNR (dB) ↑ | 34.4579 ± 0.0129 |
| SSIM ↑      |  0.9286 ± 0.0002 |
| LPIPS ↓     |  0.0498 ± 0.0003 |
| DISTS ↓     |  0.1238 ± 0.0009 |

## Environment

The experiments were conducted using:

* Python 3.10.12
* PyTorch 2.11.0
* CUDA 12.8
* NVIDIA GeForce RTX 5070 Ti Laptop GPU

Required Python packages are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

## Repository Contents

The repository provides:

* GHM-Mamba model implementation
* Training and evaluation notebook
* Pretrained model checkpoint
* Quantitative experimental results
* Run-to-run variability results
* Computational complexity analysis
* E7/E8 spatial error-map analysis

## Training and Evaluation

The main notebook contains the workflow for:

1. Creating/loading `Module.py`
2. Training GHM-Mamba
3. Evaluating the trained model

Dataset and output paths should be set to the corresponding local directories before execution.

## Pretrained Model

The pretrained EMA generator checkpoint can be used for evaluation without retraining the model.

The evaluation code loads:

```text
best_ema_generator.pt
```

and evaluates the model on the test set at `σ = 25`.

## Reproducibility

The repository includes the implementation and experimental files used to reproduce the reported GHM-Mamba experiments, including the training/evaluation pipeline, run-to-run results, computational-complexity analysis, and spatial error analysis.

## Citation

Citation information will be added after publication.
