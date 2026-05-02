# 🌸 MLflow Iris Dataset — Model Training & Saving

## 🚀 Overview
This project demonstrates a simple end-to-end **machine learning lifecycle** using **MLflow**.  
It covers:
- Model training  
- Experiment tracking  
- Model saving and reuse  

The **Iris dataset** is used to build a classification model while showcasing reproducibility and model management.

---

## 🎯 Objective
- Train a classification model on the Iris dataset  
- Track experiments using MLflow  
- Save and reload trained models  
- Build a clean and reproducible ML pipeline  

---

## 📊 Dataset
- **Dataset:** Iris Dataset  
- **Features:** Sepal length, sepal width, petal length, petal width  
- **Target:** Flower species (Setosa, Versicolor, Virginica)

---

## 🧠 Model
- Logistic Regression (primary model)  
- Extendable to:
  - Random Forest  
  - SVM  
  - Neural Networks  

---

## 🏗️ Workflow

---

## ⚙️ Features
- 📌 MLflow experiment tracking (parameters, metrics, artifacts)  
- 📌 Model versioning and saving  
- 📌 Reproducible pipeline  
- 📌 Clean and simple structure  

---

## 🧪 Example MLflow Run

```json
{
  "model": "LogisticRegression",
  "accuracy": 0.96,
  "parameters": {
    "C": 1.0,
    "solver": "lbfgs"
  }
}
