# Multiple Pregnancy Loss Analysis – BDHS 2022

![R](https://img.shields.io/badge/R-4.2+-blue?logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?logo=github)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/muhammadsalek/Multiple-Pregnancy-Loss-Analysis-BDHS-2022)
![Stars](https://img.shields.io/github/stars/muhammadsalek/Multiple-Pregnancy-Loss-Analysis-BDHS-2022?style=social)
![Forks](https://img.shields.io/github/forks/muhammadsalek/Multiple-Pregnancy-Loss-Analysis-BDHS-2022?style=social)

---

## 🌟 Project Overview

This repository contains code and analysis for:

*"Identifying Key Determinants of Multiple Pregnancy Loss among Bangladeshi Women: A Machine Learning Approach Using BDHS 2022"*

**Goals:**

- Detect determinants of recurrent pregnancy loss in Bangladeshi women.  
- Apply **explainable ML models**: Random Forest, XGBoost, Logistic Regression, Decision Tree, Neural Network, SVM, KNN.  
- Interpret model outputs with **SHAP** and **LIME** for feature importance.  
- Visualize **spatial variation** across Bangladesh.  
- Provide actionable insights for **maternal health policy**.

---

## 🗂 Repository Structure

| File / Folder               | Description |
|-----------------------------|-------------|
| `Salek_ML_Pregnancy.R`     | Main R script: preprocessing, modelling, visualization |
| `bdhs_ml_dataset.dta`      | Anonymized dataset or example data |
| `README.md`                | Project documentation |
| `LICENSE`                  | MIT License |
| `.gitignore`               | Git ignore file |

---

## 🛠 Tech Stack

![R](https://img.shields.io/badge/R-4.2+-blue?logo=r&logoColor=white)
![ML](https://img.shields.io/badge/Machine_Learning-F50057?logo=scikitlearn&logoColor=white)
![Data](https://img.shields.io/badge/Data-Orange?logo=postgresql&logoColor=white)
![Visualization](https://img.shields.io/badge/Visualization-ff69b4?logo=tableau&logoColor=white)
![Spatial](https://img.shields.io/badge/Spatial_Analysis-00BFFF?logo=mapbox&logoColor=white)

---

## ⚡ Installation & Requirements

- **R version:** >= 4.2  
- **Required Packages:**  

```r
install.packages(c("tidyverse","caret","Boruta","xgboost","randomForest","e1071","shapley","lime","sf"))
