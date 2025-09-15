##  Project Overview
**Author: Alan Royce Gabriel BS22B001**

This project investigates the effectiveness of **Gaussian Mixture Model (GMM)-based synthetic data generation** for handling class imbalance in a binary classification task (fraud detection). The dataset is highly imbalanced, with the minority class representing fraudulent transactions.

The primary objective is to compare the **baseline model** against models trained with **GMM oversampling** and its variations, and to evaluate whether synthetic data improves minority class detection.

---

## Methodology

1. **Baseline Model**

   * Logistic Regression trained on the original dataset.
   * Serves as the reference point.

2. **GMM Oversampling**

   * Minority class expanded using Gaussian Mixture Models.
   * Assumes minority distribution follows Gaussian mixtures.

3. **GMM + Threshold Tuning**

   * Applied optimal probability threshold (instead of 0.5) to maximize F1.

4. **GMM + Threshold + Regularization (C)**

   * Logistic Regression regularization hyperparameter tuned jointly with GMM and threshold.

5. **Evenly Balanced Dataset**

   * Equal samples from both classes to test the effect of aggressive balancing.

---

## Results Summary

| Model                   | Precision (Fraud=1) | Recall (Fraud=1) | F1-score (Fraud=1) |
| ----------------------- | ------------------- | ---------------- | ------------------ |
| **Baseline**            | 0.85                | 0.77             | **0.81**           |
| GMM                     | 0.07                | **0.88**         | 0.12               |
| GMM + Threshold         | 0.54                | 0.67             | 0.60               |
| GMM + Threshold + C     | 0.76                | 0.79             | 0.78               |
| Balanced Dataset (Even) | 0.06                | 0.86             | 0.10               |

* **Baseline** performed best overall (F1 = 0.81).
* **GMM alone** achieved very high recall but precision collapsed.
* **Threshold tuning + C** improved balance (F1 = 0.78), but still below baseline.
* **Even balancing** replicated GMM’s issues with precision collapse.

---

## Key Insights

* GMM oversampling increases recall but introduces **synthetic noise**, leading to false positives.
* Oversampling alone is **not reliable** for fraud detection.
* Tuning thresholds and regularization mitigates the issue but still does not beat the baseline.
* The Gaussian assumption of GMM is unsuitable for the **sparse, irregular distribution** of fraudulent data.

---

## Conclusion

* **Recommendation**: Do not use GMM-based oversampling as the primary method for class imbalance in this task.
* The **baseline model already outperforms GMM** in balanced performance.
* If oversampling is required, consider **SMOTE variants (Borderline-SMOTE, ADASYN)** or **ensemble methods (EasyEnsemble, BalancedBagging)**, which preserve local structure better.

---

## Repository Contents

* `bs22b001_A4.ipynb` — Jupyter Notebook with full code, experiments, and evaluations.



