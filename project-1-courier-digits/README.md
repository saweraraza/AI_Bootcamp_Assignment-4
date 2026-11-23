# Project 1: Courier Company — Little Data, Big Model

A PyTorch diagnostic and regularization study proving how learning curves, Dropout, Weight Decay (`AdamW`), and Early Stopping prevent a high-capacity neural network from memorizing a small dataset ($N = 200$).

---

## Table of Contents
1. [Problem Statement & Objectives](#problem-statement--objectives)
2. [Implementation Architecture](#implementation-architecture)
3. [VS Code Setup & Execution](#vs-code-setup--execution)
4. [Experimental Results](#experimental-results)
5. [Engineering Decision & Next Steps](#engineering-decision--next-steps)
6. [Limitations](#limitations)

---

## Problem Statement & Objectives
A courier company has only **200 labeled pincode digit images** and needs an automated digit reader. Training a large network on small data often produces near-zero training error while failing on unseen mail.

**Key Objectives:**
1. Split the dataset using a fixed random seed into **200 training**, **798 validation**, and **799 test** images.
2. Train three neural network architectures of increasing complexity and regularization for 300 epochs.
3. Measure both training and validation loss in evaluation mode (`model.eval()`, `torch.no_grad()`) after every epoch.
4. Generate a side-by-side learning curve visualization (`curves.png`) demonstrating **underfitting**, **overfitting**, and a **regularized good fit**.
5. Save the best validation checkpoint (`best_reg.pt`) using early stopping and evaluate final test accuracy.

---

## Implementation Architecture
* **Dataset:** `sklearn.datasets.load_digits()` ($1,797$ grayscale $8 \times 8$ images flattened to 64 features, scaled to $[0, 1]$).
* **Reproducibility:** Fixed NumPy and PyTorch seeds (`SEED = 42`) with stratified train/validation/test splitting.
* **Three Models Compared:**
  1. **Small (`nn.Linear(64, 10)`):** Single linear layer trained with `Adam(lr=0.001)`.
  2. **Big (`64 → 256 → 256 → 10`):** Two-hidden-layer ReLU MLP trained with `Adam(lr=0.001)` and no regularization.
  3. **Big + Dropout + Weight Decay:** Two-hidden-layer ReLU MLP with `nn.Dropout(0.3)`, `AdamW(lr=0.001, weight_decay=0.05)`, best-checkpoint saving, and **Early Stopping (`patience=30`)**.

---

## VS Code Setup & Execution

### 1. Navigate to the Project Folder
```bash
cd project-1-courier-digits
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training & Curve Generation
```bash
python main.py
```

### Generated Artifacts
* `curves.png` — Side-by-side plot of training vs. validation loss across all three models.
* `best_small.pt`, `best_big.pt`, `best_reg.pt` — Saved PyTorch model weights at each model's lowest validation loss epoch.

---

## Experimental Results

| Model | Curve Diagnosis | Final Train Loss | Best Validation Loss | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Small (Linear)** | **Underfitting** (both losses remain high) | `1.767` | `1.800` (epoch 300) | `75.1%` (`0.751`) |
| **Big (2×256 MLP)** | **Overfitting** (train hits `0.001`, val climbs after ep 84) | `0.001` | `0.248` (epoch 84) | `93.4%` (`0.934`) |
| **Big + Dropout + Decay** | **Healthy Fit** (stops early at ep 187, best at ep 157) | `0.006` | **`0.234`** (epoch 157) | **`93.7%`** (`0.937`) |

---

## Engineering Decision & Next Steps
* **Which Model Ships:** Ship the **regularized model (`best_reg.pt`)**. It achieves the lowest validation loss (`0.234`), prevents memorization via Dropout and Weight Decay, and stops itself 113 epochs early at peak generalization.
* **What Would Help Most Next:** **Collecting more labeled training data.** With only 200 training images (~20 samples per digit class), dataset size—not model architecture—is the primary bottleneck holding back test accuracy.

---

## Limitations
1. **Strict Input Format:** Expects centered, low-resolution $8 \times 8$ grayscale digit matrices scaled to $[0, 1]$; full envelope photos or uncropped handwritten digits will fail without an upstream segmentation pipeline.
2. **Spatial Shift Sensitivity:** Trained without random pixel-shift augmentation, so slight misalignment in a physical scanner can degrade accuracy.
3. **Confusable Digit Pairs:** Visually similar digits (such as `9` misclassified as `7` or `5`) remain error-prone due to the small sample count per class.