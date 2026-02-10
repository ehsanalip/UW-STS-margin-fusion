

# Multimodal Fusion Training Pipeline for STS post-surgical margin prediction.

This repository provides a unified PyTorch training and evaluation pipeline for **multimodal models** using **image data (DICOM-derived)** and **tabular clinical features**.  
It supports three common fusion strategies:

- **Early Fusion**
- **Intermediate Fusion**
- **Late Fusion**

Please refer to the paper for required features and files.

---

## Supported Fusion Types

### 1. Early Fusion
Image and clinical features are concatenated **before** the final prediction layer.

### 2. Intermediate Fusion
Image and clinical features are processed by separate encoders and fused at an intermediate latent representation.

### 3. Late Fusion
Independent image and clinical predictions are generated first and fused at the decision level.

---

## Repository Structure

```text
.
├── train_fusion.py        # Unified training & testing script
├── helper.py              # Shared helper functions (models, training utils)
├── README.md              # This file
├── Model Weights          # This folder contains the trained model weights for each fusion strategy
