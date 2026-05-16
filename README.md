# Adaptive Instance-Level Reweighting for Imbalanced Classification

## Overview

This project implements and evaluates the research paper:

> “A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty”

The objective is to study adaptive instance-level reweighting for imbalanced classification problems and compare its performance against standard machine learning baselines.

The experiments were conducted on two real-world imbalanced datasets:
- Credit Card Fraud Detection Dataset
- Mammography Dataset

In addition to reproducing the original proposed method, two extensions were also explored:
- Weighted-loss extension
- Ensemble learning extension

---

## Baseline Models

The following machine learning algorithms were used for comparison:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest (RF)

---

## Proposed Variants

- Proposed Adaptive Reweighting
- Weighted-Loss Extension
- Ensemble Extension

---

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

PR-AUC was treated as an important metric for highly imbalanced datasets.

---

## Main Observations

- Weighted-loss extension achieved strong PR-AUC performance on the highly imbalanced Credit Card Fraud dataset.
- Ensemble extension achieved the best overall performance on the Mammography dataset.
- Adaptive instance-level weighting achieved competitive minority-class detection performance without explicit oversampling or undersampling.

---

## Repository Structure

```text
DA5400_Final_Project/
│
├── figures/
├── datasets/
├── src/
├── credit_card_experiments.ipynb
├── mammography_experiments.ipynb
├── final_report.pdf
└── README.md
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Author

Ankit Gangwar  
Ph.D Scholar, IIT Madras  
mm25d950@smail.iitm.ac.in
