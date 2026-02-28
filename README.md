# Multiple Pregnancy Loss Analysis – BDHS 2022

![R](https://img.shields.io/badge/R-4.2+-blue?logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?logo=github)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/muhammadsalek/Multiple-Pregnancy-Loss-Analysis-BDHS-2022)
![GitHub Repo Size](https://img.shields.io/github/repo-size/muhammadsalek/Multiple-Pregnancy-Loss-Analysis-BDHS-2022)

**Author:** Md Jamal Uddin, Md Salek Miah  
**Email:** [saleksta@gmail.com](mailto:saleksta@gmail.com)  

---

## 🎯 Project Overview

This repository contains the code and analysis for:

*"Identifying Key Determinants of Multiple Pregnancy Loss among Bangladeshi Women: A Machine Learning Approach Using BDHS 2022"*

**Key Objectives:**

- Identify determinants of recurrent pregnancy loss using nationally representative data.  
- Apply **explainable machine learning models**: Random Forest, XGBoost, Logistic Regression, Decision Tree, Neural Network, SVM, KNN.  
- Interpret model predictions with **SHAP** and **LIME**.  
- Visualize **spatial variation** of pregnancy loss prevalence across Bangladesh.  

---

## 📁 Repository Structure

| File / Folder               | Description |
|-----------------------------|-------------|
| `Salek_ML_Pregnancy.R`     | Main R script: preprocessing, modelling, visualization |
| `bdhs_ml_dataset.dta`      | DHS dataset (anonymized or example) |
| `README.md`                | Project documentation |
| `LICENSE`                  | MIT License |
| `.gitignore`               | Files to exclude from Git tracking |

---

## 🛠 Requirements

- **R version:** >= 4.2  
- **R Packages:**  
`tidyverse`, `caret`, `Boruta`, `xgboost`, `randomForest`, `e1071`, `shapley`, `lime`, `sf`  

> Install missing packages:

```r
install.packages(c("tidyverse","caret","Boruta","xgboost","randomForest","e1071","shapley","lime","sf"))
