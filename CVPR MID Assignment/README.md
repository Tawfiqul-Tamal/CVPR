# 🌿 Plant Disease Classification using Custom CNN
A custom Convolutional Neural Network (CNN) built with PyTorch to classify plant leaf images into **38 categories** (healthy vs. diseased) across **14 crop species**, developed as part of the **CVPR Mid Assignment (Summer 25-26)**.

---
## 📌 Project Overview

Crop diseases are among the leading causes of yield loss in agriculture, and manual scouting is slow, subjective, and doesn't scale. This project builds an end-to-end deep learning pipeline — from data loading and augmentation to training, evaluation, and model persistence — to automatically detect plant diseases from leaf images.

| | |
|---|---|
| **Course** | CVPR — Mid Assignment (Summer 25-26) |
| **Student Name** | Md. Tawfiqul Islam Tamal |
| **Student ID** | 23-52978-3 |
| **Institution** | American International University-Bangladesh (AIUB) |
| **Framework** | PyTorch |
| **Task** | Multi-class image classification (38 classes) |
| **Domain** | Agriculture / Precision Farming |

---
## 📂 Dataset

**New Plant Diseases Dataset (Augmented)** — an augmented, re-organized version of the widely-used PlantVillage dataset.

- 🔗 **Source:** [Kaggle — vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 🖼️ **Total images:** ~87,000
- 🏷️ **Classes:** 38 (healthy + disease categories across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- 📁 **Structure:** Pre-organized in `ImageFolder` format (one sub-folder per class)

### Why this dataset?
- Multi-class, fine-grained classification challenge suitable for a from-scratch CNN
- Directly relevant to a planned follow-up research paper on agriculture-focused explainable ML
- Clean, well-documented, and widely benchmarked in plant disease recognition literature

---

## 🧠 Model Architecture

A custom **4-block CNN** designed to balance capacity and efficiency:

- 4 convolutional blocks (32 → 64 → 128 → 256 channels)
- BatchNorm + ReLU after every convolution for stable training
- Dropout2d (0.1 → 0.4, increasing with depth) to control overfitting
- AdaptiveAvgPool2d(4×4) to keep the parameter count — and saved model size — small
- Fully connected classifier head with Dropout(0.5) before the final 38-way output layer

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr = 1e-3, weight_decay = 1e-4) |
| LR Scheduler | CosineAnnealingLR (eta_min = 1e-5) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Input Size | 128 × 128 |
| Epochs | 20 |

---

## 🔀 Data Split

| Split | Source | Approx. Images | Purpose |
|---|---|---|---|
| Train | 90% of `train/` folder | ~63,265 | Model training with augmentation |
| Validation | 10% of `train/` folder | ~7,030 | Per-epoch monitoring, LR scheduling |
| Test | 100% of `valid/` folder | ~17,572 | Final, fully held-out evaluation |

The test set is never used during training or hyperparameter tuning, ensuring an unbiased final performance estimate.

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **99.04%** |
| Weighted Precision | 0.9905 |
| Weighted Recall | 0.9904 |
| Weighted F1-score | 0.9904 |
| Best Validation Accuracy (during training) | 99.17% (Epoch 18) |
| Trained Model Size | 10.34 MB |

**Best-performing classes (F1-score):** Cherry Powdery Mildew (1.000), Grape Leaf Blight — Isariopsis (1.000), Strawberry healthy (1.000), Grape Esca/Black Measles (0.999), Grape Black Rot (0.999)

**Lowest-performing classes (F1-score):** Corn Cercospora/Gray Leaf Spot (0.956), Tomato Late Blight (0.963), Tomato Early Blight (0.964), Corn Northern Leaf Blight (0.967), Tomato Target Spot (0.976) — still strong, but these visually similar same-crop diseases are the model's main source of confusion (see confusion matrix in the notebook).

**Training curves and confusion matrix:** see the notebook's Section 7 output cells for the full accuracy/loss curves and the row-normalized (%) confusion matrix.

---

## 📁 Repository Structure

```
├── CNN_<StudentID>.ipynb        # Full notebook: data loading, training, evaluation
├── plant_disease_cnn_final.pth  # Trained model weights (<20MB)
├── README.md                    # This file

```

---

## ⚙️ How to Run

1. Open the notebook in **Google Colab** or **Kaggle Notebooks**
2. Set the hardware accelerator to **GPU (T4)**
3. Set up Kaggle API credentials (`kaggle.json`) for automatic dataset download via `kagglehub`
4. Run all cells top to bottom
5. Trained weights will be saved as `plant_disease_cnn_final.pth`

### Loading the trained model

```python
import torch
from model import PlantDiseaseCNN  # class defined in the notebook

model = PlantDiseaseCNN(num_classes=38)
model.load_state_dict(torch.load("plant_disease_cnn_final.pth", map_location="cpu"))
model.eval()
```

---

## 🔮 Future Work

- Compare against transfer learning (ResNet / EfficientNet pretrained on ImageNet)
- Add Grad-CAM / SHAP explainability to visualize disease-relevant leaf regions
- Evaluate robustness on real, uncontrolled-background field images
- Explore ensembling multiple CNN backbones for confusable disease pairs

---

## 📄 License & Attribution

Dataset: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), an augmented version of the original **PlantVillage** dataset. Used for academic/educational purposes as part of a university coursework assignment.
