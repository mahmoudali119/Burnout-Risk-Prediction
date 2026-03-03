## 🔥 Burnout Risk Prediction

This project builds a machine learning system to predict employee burnout risk (Low, Medium, High) based on behavioral and work-related features.

### 📌 Problem

Employee burnout is a serious issue affecting productivity and well-being.
The goal is to predict burnout risk levels using structured behavioral data.

---

### 📊 Dataset

* 1800 samples
* Target: `burnout_risk` (Low, Medium, High)
* Strong class imbalance (High class is rare)

Special attention was given to:

* Detecting and removing **target leakage**
* Handling severe **class imbalance**

---

### ⚠️ Key Challenges

* `burnout_score` was highly correlated with the target and caused target leakage.
* Severe class imbalance (High class extremely underrepresented).
* Standard accuracy metric was misleading.

---

### 🧠 Approach

1. Data Cleaning & EDA
2. Leakage detection and removal
3. Proper target encoding using `LabelEncoder`
4. Stratified train-test split
5. Preprocessing using `Pipeline` and `ColumnTransformer`
6. Class imbalance handling using:

   * SMOTE (inside pipeline)
   * Class weighting
7. Threshold tuning to improve Macro F1

---

### 📈 Results

* Baseline Logistic Regression: Macro F1 ≈ 0.56
* Random Forest Baseline: Macro F1 ≈ 0.60
* Logistic Regression + SMOTE: Macro F1 ≈ 0.61
* Logistic + SMOTE + Threshold tuning: **Macro F1 ≈ 0.69**

Threshold tuning significantly improved performance on the rare High-risk class.

---

### 🏗 Final Model

* Logistic Regression with SMOTE
* Custom threshold for High-risk detection
* Saved model + label encoder for deployment

---

### 🚀 Key Learnings

* Perfect accuracy can indicate data leakage.
* Macro F1 is essential for imbalanced classification.
* Threshold tuning can significantly improve rare-class detection.
* Proper pipeline design prevents data leakage.

---
> Future Improvements:
>
> * Cross-validated threshold optimization
> * Gradient Boosting models
> * Model monitoring after deployment



