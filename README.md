# Applying Classification Models to Real-World Data: Linear Regression, Logistic Regression, and SVM in Python

> **Machine Learning Foundations — Lab 4**
> Purdue University Cybersecurity Workforce Certification Training Program

---

## 📌 Project Overview

This project demonstrates the application of three fundamental supervised machine learning algorithms to real-world datasets using Python and Scikit-Learn:

- **Linear Regression** — predicting continuous housing values from the Boston Housing dataset
- **Logistic Regression** — classifying diabetes outcomes from patient clinical data
- **Support Vector Machine (SVM)** — binary classification with both linear and RBF kernels, including hyperparameter tuning via Grid Search

All code was developed and run in **Google Colab**.

---

## 🗂️ Repository Structure

```
├── ML_Lab4.ipynb          # Main Jupyter notebook with all tasks
├── data1.csv              # Boston Housing dataset (reshaped for linear regression)
├── diabetes.csv           # Pima Indians Diabetes dataset
└── README.md
```

---

## 🧪 Tasks

### Task 1 — Linear Regression (Boston Housing Dataset)
- Loaded and reshaped a non-standard CSV layout using NumPy
- Split data 60/40 (train/test) with `train_test_split`
- Trained `sklearn.linear_model.LinearRegression`
- Evaluated with **Variance Score (R²)** and **Mean Squared Error (MSE)**

### Task 2 — Logistic Regression (Diabetes Dataset)
- Loaded the Pima Indians Diabetes dataset with named feature columns
- Selected 7 clinical features; split data 70/30
- Trained `LogisticRegression` with `max_iter=1000`
- Evaluated with a **confusion matrix** (visualized via seaborn heatmap) and **classification report** (precision, recall, F1-score)

### Task 3 — Support Vector Machine (Diabetes Dataset)
- **Linear Kernel:** Trained `SVC(kernel='linear')` and compared results with logistic regression
- **RBF Kernel + Grid Search (Accuracy):** Used `GridSearchCV` with `StratifiedShuffleSplit` to search over `C` and `gamma` ranges, optimizing for accuracy
- **RBF Kernel + Grid Search (Recall):** Re-ran grid search using `recall_macro` scoring to prioritize minimizing false negatives — a clinically meaningful choice for disease detection

---

## 📊 Key Results

| Model | Metric | Value |
|---|---|---|
| Linear Regression | Variance Score (R²) | ~0.72 |
| Linear Regression | MSE | ~25.2 |
| Logistic Regression | Accuracy | ~74% |
| SVM (Linear Kernel) | Accuracy | ~74% |
| SVM (RBF + Grid Search, Accuracy) | Cross-val Score | ~76% |
| SVM (RBF + Grid Search, Recall) | Cross-val Score | ~76% |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Core programming language |
| Scikit-Learn | ML models, preprocessing, evaluation |
| NumPy | Data reshaping and numerical computation |
| Pandas | Data loading and feature selection |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix heatmaps |
| Google Colab | Development environment |

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ml-lab4-classification.git
   cd ml-lab4-classification
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn numpy pandas matplotlib seaborn
   ```

3. Open the notebook:
   ```bash
   jupyter notebook ML_Lab4.ipynb
   ```
   Or upload it directly to [Google Colab](https://colab.google/).

4. Ensure `data1.csv` and `diabetes.csv` are in the same directory as the notebook (or update the file paths).

---

## 📚 Datasets

- **data1.csv** — A version of the Boston Housing dataset, stored in an interleaved row format that requires preprocessing before use.
- **diabetes.csv** — The [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), containing 8 clinical features used to predict diabetes onset.

---

## 📖 Related Article

Read the full write-up on Medium:
**[Applying Classification Models to Real-World Data: Linear Regression, Logistic Regression, and SVM in Python Using Scikit-Learn](#)**
*(Replace `#` with your Medium article URL once published)*


## 🎓 Course Context

This lab is part of the **Machine Learning Foundations** module within Purdue University's **Cybersecurity Workforce Certification Training Program**. The program emphasizes hands-on Python implementation of core ML concepts.

---

