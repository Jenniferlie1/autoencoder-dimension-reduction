# Autoencoder for Dimension Reduction — Ships vs Helicopter
Image dimension reduction using a Convolutional Autoencoder on the Overhead-MNIST dataset (ships & helicopters), with baseline and improved/tuned architectures.
## Overview
- Grayscale images resized to 28×28, normalized to [0,1], flattened to 784-dim vectors
- Stratified train/val/test split (80/10/10)
- Convolutional Autoencoder: Encoder compresses images into a latent representation, Decoder reconstructs the original image
## Workflow
1. Loading, Scaling & Split — grayscale conversion, resizing, normalization, stratified split
2. Baseline Autoencoder — Conv2D encoder/decoder with latent_dim=128, trained with Adam + MSE loss
3. Improved Autoencoder & Hyperparameter Tuning — architecture refinement and tuning over the baseline
4. Evaluation — reconstruction quality comparison between baseline and improved models
## Tech Stack
Python TensorFlow/Keras scikit-learn NumPy Matplotlib scikit-image
## Data
Overhead-MNIST subset (ship / helicopter classes) — loaded from Google Drive in the notebook (/content/drive/MyDrive/overhead); not included in this repo
