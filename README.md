# Lightweight Deep Learning Models for Real-Time Waste Classification

A comparative study of lightweight deep learning architectures for efficient and real-time waste image classification. The project evaluates model accuracy, computational efficiency, inference speed, memory requirements, and explainability.

## Overview

Automated waste classification can support intelligent recycling and smart waste-management systems. However, deploying deep learning models on edge devices requires a balance between classification performance and computational efficiency.

This project investigates and compares six lightweight deep learning models for multi-class waste classification:

- EfficientViT
- GhostNet
- ShuffleNetV2
- MobileNetV3-Large
- EfficientNet-Lite0
- SqueezeNet

The models are evaluated not only by classification performance but also by their computational cost and real-time inference characteristics.

## Objectives

The main objectives of this study are to:

- Compare lightweight CNN and vision-transformer-based architectures.
- Evaluate classification performance across multiple waste categories.
- Measure model size and computational complexity.
- Analyze CPU inference latency and FPS.
- Evaluate GPU memory consumption.
- Compare model robustness using 5-fold cross-validation.
- Investigate model predictions using Grad-CAM and Grad-CAM++.
- Identify a suitable architecture for real-time and resource-constrained deployment.

## Waste Categories

The classification task contains six waste categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Models Evaluated

| Model | Type |
|---|---|
| EfficientViT | Lightweight Vision Transformer |
| GhostNet | Lightweight CNN |
| ShuffleNetV2 | Lightweight CNN |
| MobileNetV3-Large | Lightweight CNN |
| EfficientNet-Lite0 | Efficient CNN |
| SqueezeNet | Compact CNN |

## Evaluation Metrics

The models are evaluated using:

### Classification Performance

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

### Computational Efficiency

- Number of parameters
- GFLOPs
- Model size (MB)
- CPU inference latency
- 95th-percentile CPU latency
- Frames per second (FPS)
- GPU memory consumption

## Experimental Methodology

The project uses a common training and evaluation pipeline for all architectures to enable a fair comparison.

The evaluation includes both a held-out test set and 5-fold cross-validation.

The cross-validation experiment uses:

- CV pool: 2,147 images
- Held-out test set: 380 images
- 5-fold cross-validation

The notebook also performs efficiency measurements using repeated inference runs to estimate CPU latency and FPS.

## Main Results

The initial evaluation produced the following results:

| Model | Accuracy | F1 | ROC-AUC | Params (M) | Size (MB) | CPU ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EfficientViT** | **0.9526** | **0.9476** | **0.9948** | 2.137 | 8.64 | **16.30** | **61.3** |
| GhostNet | 0.9289 | 0.9157 | 0.9945 | 3.909 | 15.88 | 36.73 | 27.2 |
| ShuffleNetV2 | 0.9105 | 0.8997 | 0.9900 | 1.260 | 5.20 | 24.61 | 40.6 |
| MobileNetV3-Large | 0.9158 | 0.8997 | 0.9899 | 4.210 | 17.03 | 24.92 | 40.1 |
| EfficientNet-Lite0 | 0.8921 | 0.8733 | 0.9905 | 3.379 | 13.77 | 32.27 | 31.0 |
| SqueezeNet | 0.8605 | 0.8401 | 0.9845 | 0.726 | 2.92 | 24.79 | 40.3 |

EfficientViT achieved the strongest overall performance in the reported evaluation, with an F1-score of **0.9476**, ROC-AUC of **0.9948**, and approximately **61.3 FPS** CPU inference speed. :contentReference[oaicite:1]{index=1}

## 5-Fold Cross-Validation

The 5-fold experiment further evaluated the stability of the models.

| Model | Test F1 Mean | Test F1 Std | Test Accuracy Mean |
|---|---:|---:|---:|
| **EfficientViT** | **0.9187** | 0.0148 | **0.9258** |
| GhostNet | 0.9100 | 0.0133 | 0.9237 |
| ShuffleNetV2 | 0.9009 | 0.0126 | 0.9111 |
| MobileNetV3-Large | 0.8886 | 0.0130 | 0.9005 |
| EfficientNet-Lite0 | 0.8692 | 0.0191 | 0.8832 |
| SqueezeNet | 0.8446 | 0.0164 | 0.8574 |

EfficientViT achieved the highest mean test F1-score of approximately **0.919 ± 0.015** across the five folds. :contentReference[oaicite:2]{index=2}

## Efficiency Analysis

The project emphasizes real-time deployment rather than classification performance alone.

EfficientViT demonstrated a strong balance between accuracy and computational efficiency:

- **2.137M parameters**
- **8.64 MB model size**
- **0.202 GFLOPs**
- **16.30 ms average CPU latency**
- **61.3 FPS**
- **39.1 MB GPU memory**

This makes EfficientViT particularly promising for resource-constrained and real-time waste-classification applications. :contentReference[oaicite:3]{index=3}

## Explainability

To investigate what the models use when making predictions, the project includes visual explainability using:

- Grad-CAM
- Grad-CAM++

These techniques generate activation heatmaps that highlight image regions contributing to the predicted waste category.

The notebook also compares Grad-CAM and Grad-CAM++ visualizations for the selected model. :contentReference[oaicite:4]{index=4}

## Technologies

- Python
- PyTorch
- TorchVision
- timm
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Grad-CAM / Grad-CAM++
- Google Colab / Jupyter Notebook

## Repository Structure

```text
lightweight-deep-learning-waste-classification/
│
├── Lightweight_Deep_Learning_Models_for_Real_Time_Waste_Classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── results/
    ├── figures/
    └── tables/
