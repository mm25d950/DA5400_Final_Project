# Adaptive Instance-Level Reweighting for Imbalanced Classification

Implementation and evaluation of adaptive instance-level reweighting for imbalanced classification on Credit Card Fraud and Mammography datasets.

---

## Overview

This project implements and experimentally evaluates the research paper:

> “A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty”

The objective is to improve minority-class detection using adaptive instance-level weighted sampling based on sample learning difficulty.

The framework is evaluated on:

- Credit Card Fraud Detection Dataset
- Mammography Dataset

Baseline models:

- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF)

Proposed variants:

- Adaptive Reweighting
- Weighted-Loss Extension
- Ensemble Extension

---

## Repository Structure

```text
DA5400_Final_Project/
│
├── figures/                 # Result visualizations and plots
├── code_notebooks/               # Jupyter notebook implementation
├── report/                  # Final report PDF
├── presentation/            # Presentation PDF/PPT
├── requirements.txt         # Python dependencies
├── README.md
└── LICENSE
