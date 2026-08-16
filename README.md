# ML Assignment 2 — Bank Marketing Classification

## 1. Problem Statement

The objective of this assignment is to develop and compare multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit based on demographic, financial and campaign-related attributes.

This is a binary classification problem where:

- 0 = No subscription
- 1 = Yes subscription

Five classification models were implemented from scratch and evaluated on the same dataset.

---

## 2. Dataset Description

### Dataset
Bank Marketing Dataset

### Source
UCI Machine Learning Repository / Bank Marketing Dataset

### Problem Type
Binary Classification

### Number of Instances
45,211

### Number of Features
16 input features

### Target Variable
`y`

The target indicates whether the customer subscribed to a term deposit.

The dataset contains demographic information, financial information and details related to the marketing campaign.

---

## 3. Models Implemented

The following models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The models were implemented using custom Python/NumPy-based implementations. OneHotEncoder and StandardScaler were used for preprocessing.

---

## 4. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8864 | 0.8756 | 0.5906 | 0.0955 | 0.1644 | 0.2046 |
| Decision Tree | 0.9010 | 0.8674 | 0.6200 | 0.3979 | 0.4847 | 0.4459 |
| kNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8549 | 0.8096 | 0.4062 | 0.5198 | 0.4561 | 0.3776 |
| Random Forest (Ensemble) | 0.7758 | 0.8375 | 0.3077 | 0.7325 | 0.4333 | 0.3686 |

---

## 5. Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 88.64% and the highest AUC of 0.8756. However, its recall was relatively low at 9.55%, meaning that it identified only a small proportion of positive cases.

### Decision Tree

Decision Tree achieved the highest accuracy of 90.10%, highest precision of 62.00%, highest F1-score of 48.47%, and highest MCC of 0.4459. It provided the strongest overall balanced performance among the evaluated models.

### kNN

kNN achieved an accuracy of 89.62%. Its precision, recall and F1-score were moderate. The model performed reasonably well but did not outperform the Decision Tree on the main evaluation metrics.

### Naive Bayes

Naive Bayes achieved an accuracy of 85.49%. It obtained a recall of 51.98%, which was higher than Logistic Regression, Decision Tree and kNN. However, its precision and AUC were comparatively lower.

### Random Forest

Random Forest achieved the highest recall of 73.25%, meaning it identified the largest proportion of positive cases. However, its accuracy and precision were considerably lower than the other models in this experiment.

---

## 6. Overall Winner

Based on the overall evaluation, the **Decision Tree Classifier** is the best-performing model for this dataset.

It achieved:

- Accuracy: 90.10%
- Precision: 62.00%
- Recall: 39.79%
- F1-score: 48.47%
- MCC: 0.4459
- AUC: 0.8674

Although Random Forest achieved the highest recall, Decision Tree provided the best overall balance across accuracy, precision, F1-score and MCC.

---

## 7. Streamlit Application

The project includes an interactive Streamlit application.

The application provides:

- CSV test-data upload
- Model selection dropdown
- Model predictions
- Prediction probabilities
- Accuracy
- Precision
- Recall
- F1-score
- MCC
- AUC
- Confusion matrix
- Downloadable prediction results

The application is designed to run using the supplied test dataset.

---

## 8. Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
